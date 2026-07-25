"""SSH-based job launcher — screen sessions, kill, list, tail."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import asyncssh

from dashboard.config import load_servers
from dashboard.probe import probe_all_servers

from .job_config import build_log_path, build_task_command
from .models import GpuCandidate, JobConfig, JobRecord, LaunchRecord
from .registry import add_job, get_job, update_task_status
from .scheduler import find_free_gpus, resolve_host_port

logger = logging.getLogger(__name__)

SCREEN_PREFIX = "gpu-"


async def probe_and_rank(
    config_path: str,
    preferred_servers: Optional[list[str]] = None,
    allowed_servers: Optional[set[str]] = None,
    ssh_timeout: int = 8,
) -> list[GpuCandidate]:
    """Probe all servers and return ranked free GPU candidates."""
    servers = load_servers(config_path)
    metrics = await probe_all_servers(servers, timeout=ssh_timeout)
    host_map = {s.name: (s.host, s.port) for s in servers}
    candidates = find_free_gpus(
        metrics,
        preferred_servers=preferred_servers,
        allowed_servers=allowed_servers,
    )
    return resolve_host_port(candidates, host_map)


async def launch_job(
    job: JobConfig,
    server_config_path: str,
    dry_run: bool = False,
    max_gpus: int = 999,
    preferred_servers: Optional[list[str]] = None,
    allowed_servers: Optional[set[str]] = None,
    ssh_timeout: int = 10,
) -> JobRecord:
    """Probe, assign GPUs, and launch all tasks in a job."""
    tasks_to_launch = job.tasks[:max_gpus]
    if not tasks_to_launch:
        raise ValueError("No tasks defined in job config")

    servers = load_servers(server_config_path)
    metrics = await probe_all_servers(servers, timeout=ssh_timeout)
    host_map = {s.name: (s.host, s.port) for s in servers}

    candidates = find_free_gpus(
        metrics,
        preferred_servers=preferred_servers,
        allowed_servers=allowed_servers,
    )
    candidates = resolve_host_port(candidates, host_map)

    if len(candidates) < len(tasks_to_launch):
        print(
            f"WARNING: Only {len(candidates)} free GPUs, "
            f"{len(tasks_to_launch)} tasks requested."
        )

    server_info: dict[str, tuple[str, int, Optional[str]]] = {}
    for s in servers:
        server_info[s.name] = (s.host, s.port, s.identity_file)

    records: list[LaunchRecord] = []

    for i, task in enumerate(tasks_to_launch):
        if i >= len(candidates):
            print(f"  SKIP {task.name}: no free GPU")
            records.append(LaunchRecord(
                name=task.name, server="", gpu=-1,
                screen_session="", log_file="", status="dead",
            ))
            continue

        gpu = candidates[i]
        screen_name = f"{SCREEN_PREFIX}{task.name}"
        log_file = build_log_path(
            job.defaults.log_dir, job.project, job.name, task.name,
        )
        cmd = build_task_command(job, task, log_file, server_name=gpu.server_name)

        # Inject CUDA_VISIBLE_DEVICES directly on the python command line
        # (more reliable than export through screen's bash -c)
        pid_file = os.path.join(job.defaults.log_dir, job.project, job.name, f"{task.name}.pid")
        cmd = cmd.replace("\n", f"\nCUDA_VISIBLE_DEVICES={gpu.gpu_index} ", 1)
        # Capture PID: wrap python in subshell that writes PID to file
        cmd = cmd.replace(
            f"{job.defaults.python} ",
            f"bash -c 'echo $$ > {pid_file}; exec {job.defaults.python} \"$@\"' -- ",
            1,
        )

        full_cmd = (
            f"mkdir -p {job.defaults.log_dir}/{job.project}/{job.name} && "
            f"screen -dmS {screen_name} bash -c '{cmd}'"
        )
        if job.defaults.env:
            env_str = " ".join(f"{k}={v}" for k, v in job.defaults.env.items())
            full_cmd = f"export {env_str}; {full_cmd}"

        record = LaunchRecord(
            name=task.name, server=gpu.server_name, gpu=gpu.gpu_index,
            screen_session=screen_name, log_file=log_file, status="running",
        )

        if dry_run:
            print(f"  [DRY] {task.name}")
            print(f"    {gpu.server_name}:{gpu.gpu_index}  screen={screen_name}")
            print(f"    log: {log_file}")
            print(f"    cmd: {cmd[:180]}...")
            print()
            records.append(record)
            continue

        try:
            info = server_info[gpu.server_name]
            ck: dict = {"known_hosts": None}
            if info[2]:
                ck["client_keys"] = [info[2]]
            async with asyncssh.connect(
                info[0], port=info[1], username="user", **ck,
            ) as conn:
                r = await conn.run(full_cmd, check=False)
                if r.exit_status == 0:
                    print(f"  OK  {task.name} -> {gpu.server_name}:{gpu.gpu_index} [{screen_name}]")
                else:
                    record.status = "dead"
                    print(f"  FAIL {task.name}: {r.stderr}")
        except Exception as exc:
            record.status = "dead"
            print(f"  FAIL {task.name}: {exc}")

        records.append(record)

    job_record = JobRecord(
        name=job.name, project=job.project, config_path="", tasks=records,
    )
    if not dry_run:
        add_job(job_record)

    return job_record


async def kill_job(
    job_name: str,
    task_name: Optional[str] = None,
    server_config_path: Optional[str] = None,
    ssh_timeout: int = 8,
) -> bool:
    """Kill a job or specific task by terminating its screen session."""
    job = get_job(job_name)
    if not job:
        print(f"Job '{job_name}' not found in registry.")
        return False

    if not server_config_path:
        for t in job.tasks:
            if task_name and t.name != task_name:
                continue
            update_task_status(job_name, t.name, "killed")
        return True

    servers = load_servers(server_config_path)
    host_map = {s.name: (s.host, s.port, s.identity_file) for s in servers}
    all_ok = True

    for t in job.tasks:
        if task_name and t.name != task_name:
            continue
        if t.status in ("done", "dead", "killed"):
            continue
        info = host_map.get(t.server)
        if not info:
            print(f"  FAIL {t.name}: server '{t.server}' unknown")
            all_ok = False
            continue
        try:
            ck: dict = {"known_hosts": None}
            if info[2]:
                ck["client_keys"] = [info[2]]
            async with asyncssh.connect(
                info[0], port=info[1], username="user", **ck,
            ) as conn:
                await conn.run(f"screen -S {t.screen_session} -X quit", check=False)
                print(f"  OK  Killed {t.name} on {t.server}")
                update_task_status(job_name, t.name, "killed")
        except Exception as exc:
            print(f"  FAIL {t.name}: {exc}")
            all_ok = False

    return all_ok


async def tail_log(
    job_name: str,
    task_name: Optional[str] = None,
    lines: int = 20,
    server_config_path: Optional[str] = None,
    ssh_timeout: int = 8,
) -> str:
    """Tail log file(s) for a job via SSH."""
    job = get_job(job_name)
    if not job:
        return f"Job '{job_name}' not found in registry."

    servers = load_servers(server_config_path) if server_config_path else []
    host_map = {s.name: (s.host, s.port, s.identity_file) for s in servers}
    parts: list[str] = []

    for t in job.tasks:
        if task_name and t.name != task_name:
            continue
        if not t.log_file:
            parts.append(f"=== {t.name}: no log file ===")
            continue
        info = host_map.get(t.server)
        if not info:
            parts.append(f"=== {t.name}: server unknown ===")
            continue
        try:
            ck: dict = {"known_hosts": None}
            if info[2]:
                ck["client_keys"] = [info[2]]
            async with asyncssh.connect(
                info[0], port=info[1], username="user", **ck,
            ) as conn:
                r = await conn.run(f"tail -n {lines} {t.log_file}", check=False)
                parts.append(
                    f"=== {t.name} ({t.server}:{t.gpu}) [{t.status}] ===\n"
                    f"{r.stdout or '(empty)'}"
                )
        except Exception as exc:
            parts.append(f"=== {t.name}: {exc} ===")

    return "\n\n".join(parts)
