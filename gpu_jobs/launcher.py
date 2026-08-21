"""SSH-based job launcher — screen sessions, kill, list, tail."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
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


# ── Pre-flight check ────────────────────────────────────────────────────────


@dataclass
class PreflightResult:
    """Result of a pre-flight check on a candidate server."""
    ok: bool
    errors: list[str]
    # Diagnosed per-server paths (after remap), used to skip redundant checks
    log_dir_parent: str = ""


async def _preflight_for_server(
    host: str,
    port: int,
    identity_file: str | None,
    work_dir: str,
    python_path: str,
    log_dir_parent: str,
    ssh_timeout: int = 8,
) -> PreflightResult:
    """Run a single pre-flight SSH check on a server.

    Checks that:
      1. ``work_dir`` exists and is a directory
      2. ``python_path`` exists and is executable
      3. The parent of ``log_dir_parent`` (e.g. ``/tmp/gpu-jobs``) either exists
         & is writable, or its parent is writable so ``mkdir -p`` will work
      4. At least 10 MB of free space on the filesystem hosting the log dir
    """
    errors: list[str] = []
    # Escape paths for safe shell embedding
    check_script = (
        f"test -d '{work_dir}' || echo 'ERR:work_dir_missing:{work_dir}';"
        f"test -x '{python_path}' || echo 'ERR:python_missing:{python_path}';"
        # Try mkdir -p on the target log directory — this is what screen launch does.
        f"mkdir -p '{log_dir_parent}' 2>/dev/null;"
        f"if [ ! -d '{log_dir_parent}' ]; then echo 'ERR:log_dir_failed:{log_dir_parent}';"
        # Only check disk space if log dir exists (avoids redundant df failure)
        f"elif ! df -BM --output=avail '{log_dir_parent}' 2>/dev/null | tail -1 | grep -q .; then echo 'ERR:df_failed:{log_dir_parent}';"
        f"fi"
    )

    try:
        ck: dict = {"known_hosts": None}
        if identity_file:
            ck["client_keys"] = [identity_file]
        async with asyncssh.connect(
            host, port=port, username="user", **ck,
        ) as conn:
            result = await asyncio.wait_for(
                conn.run(check_script, check=False),
                timeout=ssh_timeout,
            )
            stdout = (result.stdout or "").strip()

            if result.exit_status != 0 and "ERR:" not in stdout:
                stderr = (result.stderr or "").strip()
                errors.append(f"preflight exited {result.exit_status}: {stderr[:200]}")
                return PreflightResult(ok=False, errors=errors)

            # Parse ERR: lines from stdout
            for line in stdout.splitlines():
                if line.startswith("ERR:"):
                    parts = line.split(":", 2)
                    code = parts[1] if len(parts) > 1 else "unknown"
                    detail = parts[2] if len(parts) > 2 else line
                    if code == "work_dir_missing":
                        errors.append(f"work_dir does not exist: {detail}")
                    elif code == "python_missing":
                        errors.append(f"python binary not found: {detail}")
                    elif code == "log_dir_failed":
                        errors.append(f"cannot create log directory: {detail}")
                    elif code == "df_failed":
                        errors.append(f"cannot check disk space on log dir: {detail}")
                    else:
                        errors.append(f"preflight: {line}")

            return PreflightResult(ok=len(errors) == 0, errors=errors)

    except (OSError, asyncssh.Error, asyncio.TimeoutError) as exc:
        return PreflightResult(ok=False, errors=[f"SSH failed: {exc}"])


async def _run_preflight_checks(
    candidates: list[GpuCandidate],
    server_info: dict[str, tuple[str, int, str | None]],
    work_dir: str,
    python_path: str,
    log_dir: str,
    project: str,
    job_name: str,
    path_remap_rules: list,
    ssh_timeout: int = 8,
) -> dict[str, PreflightResult]:
    """Run pre-flight checks on each unique server among candidates, in parallel.

    Applies per-server path remapping so that non-primary servers are checked
    with their actual (remapped) paths.

    Returns a dict mapping ``server_name`` → ``PreflightResult``.
    """
    seen: set[str] = set()
    tasks: dict[str, asyncio.Task[PreflightResult]] = {}

    for c in candidates:
        if c.server_name in seen:
            continue
        seen.add(c.server_name)
        info = server_info.get(c.server_name)
        if not info:
            continue
        host, port, id_file = info

        # Apply path remap rules for this server (same logic as build_task_command)
        wd = _apply_path_remap(work_dir, c.server_name, path_remap_rules)
        py = _apply_path_remap(python_path, c.server_name, path_remap_rules)
        ld = _apply_path_remap(log_dir, c.server_name, path_remap_rules)
        log_dir_parent = os.path.join(ld, project, job_name)

        tasks[c.server_name] = asyncio.create_task(
            _preflight_for_server(
                host=host, port=port, identity_file=id_file,
                work_dir=wd, python_path=py,
                log_dir_parent=log_dir_parent,
                ssh_timeout=ssh_timeout,
            )
        )

    results: dict[str, PreflightResult] = {}
    for name, task in tasks.items():
        results[name] = await task
    return results


def _apply_path_remap(path: str, server_name: str, rules: list) -> str:
    """Apply path remapping rules from job YAML config (first match wins)."""
    for rule in rules:
        if server_name.startswith(rule.match) and server_name not in rule.exclude:
            for find, replace in rule.map.items():
                path = path.replace(find, replace)
            break
    return path


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
    no_preflight: bool = False,
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

    # ── Pre-flight check ──────────────────────────────────────────────
    failed_servers: set[str] = set()
    if not no_preflight:
        preflight_results = await _run_preflight_checks(
            candidates=candidates,
            server_info=server_info,
            work_dir=job.defaults.work_dir,
            python_path=job.defaults.python,
            log_dir=job.defaults.log_dir,
            project=job.project,
            job_name=job.name,
            path_remap_rules=job.defaults.path_remap,
            ssh_timeout=ssh_timeout,
        )
        for name, pr in preflight_results.items():
            if not pr.ok:
                failed_servers.add(name)
        if failed_servers:
            print()
            for name in sorted(failed_servers):
                pr = preflight_results[name]
                for e in pr.errors:
                    print(f"  PREFLIGHT FAIL [{name}]: {e}")
            print()

        # Remove candidates whose server failed pre-flight
        candidates = [c for c in candidates if c.server_name not in failed_servers]

    records: list[LaunchRecord] = []

    for i, task in enumerate(tasks_to_launch):
        if i >= len(candidates):
            print(f"  SKIP {task.name}: no free GPU (preflight removed all candidates)")
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

        # Apply path remap to the launch log_dir so the mkdir + pid file target
        # the server-local path on non-primary servers (preflight already
        # remaps log_dir; match it here so the launch mkdir does not fail).
        launch_log_dir = _apply_path_remap(job.defaults.log_dir, gpu.server_name, job.defaults.path_remap)
        # Inject CUDA_VISIBLE_DEVICES directly on the python command line
        # (more reliable than export through screen's bash -c)
        pid_file = os.path.join(launch_log_dir, job.project, job.name, f"{task.name}.pid")
        cmd = cmd.replace("\n", f"\nCUDA_VISIBLE_DEVICES={gpu.gpu_index} ", 1)
        # Capture PID: wrap python in subshell that writes PID to file
        cmd = cmd.replace(
            f"{job.defaults.python} ",
            f"bash -c 'echo $$ > {pid_file}; exec {job.defaults.python} \"$@\"' -- ",
            1,
        )

        full_cmd = (
            f"mkdir -p {launch_log_dir}/{job.project}/{job.name} && "
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
