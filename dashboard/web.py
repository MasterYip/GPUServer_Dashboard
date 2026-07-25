"""FastAPI web dashboard with Server-Sent Events for real-time updates."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader
from starlette.responses import StreamingResponse

from .config import load_servers
from .models import ServerConfig, ServerMetrics
from .probe import probe_all_servers

logger = logging.getLogger(__name__)

_OUR_ENV_MARKER = "pegasusmodye"  # conda env — unique to our training processes

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    auto_reload=False,
    cache_size=1,
)


class AppState:
    """Shared mutable state for the dashboard."""

    def __init__(
        self,
        servers: list[ServerConfig],
        servers_config: str = "",
        interval: int = 10,
        ssh_timeout: int = 5,
    ):
        self.servers = servers
        self.servers_config = servers_config
        self.interval = interval
        self.ssh_timeout = ssh_timeout
        self.metrics_cache: dict[str, ServerMetrics] = {}
        self.cache_lock = asyncio.Lock()
        self._update_event = asyncio.Event()
        self.my_gpu_usage: dict = {}  # {server: {gpu: {my_util_pct, my_mem_mb, ...}}}
        self._my_usage_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Terminal launcher
# ---------------------------------------------------------------------------

_TERMINALS = [
    "gnome-terminal", "konsole", "xfce4-terminal", "mate-terminal",
    "lxterminal", "terminator", "tilix", "kitty", "alacritty",
    "wezterm", "xterm", "uxterm",
]

_TERMINAL_ARGS: dict[str, list[str]] = {
    "gnome-terminal": ["--", "bash", "-c"],
    "konsole":       ["-e", "bash", "-c"],
    "xfce4-terminal": ["-e", "bash", "-c"],
    "mate-terminal":  ["-e", "bash", "-c"],
    "lxterminal":     ["-e", "bash", "-c"],
    "terminator":     ["-e", "bash", "-c"],
    "tilix":          ["-e", "bash", "-c"],
    "kitty":          ["bash", "-c"],
    "alacritty":      ["-e", "bash", "-c"],
    "wezterm":        ["start", "--", "bash", "-c"],
    "xterm":          ["-e", "bash", "-c"],
    "uxterm":         ["-e", "bash", "-c"],
}


def _find_terminal() -> tuple[str, list[str]] | None:
    for name in _TERMINALS:
        path = shutil.which(name)
        if path:
            return path, _TERMINAL_ARGS.get(name, ["-e", "bash", "-c"])
    return None


def _launch_terminal(ssh_cmd: str) -> str | None:
    found = _find_terminal()
    if not found:
        return "No terminal emulator found. Copy the SSH command instead."
    term_path, term_args = found
    full_cmd = f"{ssh_cmd} ; echo '---'; read -p 'Press Enter to close...'"
    cmd = [term_path] + term_args + [full_cmd]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        return f"Failed to launch terminal: {exc}"
    return None


def _launch_vscode(ssh_cmd: str, server_name: str) -> str | None:
    code_path = shutil.which("code")
    if not code_path:
        return "VSCode CLI not found."
    parts = ssh_cmd.replace("ssh ", "").replace(" -p ", ":").split()
    remote = parts[0]
    code_remote = f"ssh-remote+{remote}"
    try:
        subprocess.Popen(
            [code_path, "--new-window", "--remote", code_remote],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, env=os.environ.copy(), start_new_session=True,
        )
    except OSError as exc:
        return f"Failed to launch VSCode: {exc}"
    return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config_path: str, interval: int = 10, ssh_timeout: int = 5) -> FastAPI:
    servers = load_servers(config_path)
    if not servers:
        raise SystemExit("No servers found in config.")

    server_map: dict[str, ServerConfig] = {s.name: s for s in servers}

    app = FastAPI(title="GPU Server Dashboard", version="1.0.0")
    state = AppState(
        servers=servers, servers_config=config_path,
        interval=interval, ssh_timeout=ssh_timeout,
    )

    @app.on_event("startup")
    async def _start_background_probe():
        asyncio.create_task(_probe_loop(state))
        asyncio.create_task(_my_usage_loop(state))

    # ---- Routes -----------------------------------------------------------

    @app.get("/api/my-gpu-usage")
    async def my_gpu_usage():
        """Cached per-GPU usage for our tasks.
        Returns {server: {gpu_index: {my_util_pct, my_mem_mb, total_util_pct, total_mem_mb}}}
        """
        async with state._my_usage_lock:
            return dict(state.my_gpu_usage)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        template = _jinja_env.get_template("dashboard.html")
        return template.render(
            request=request, servers=state.servers,
            server_count=len(state.servers), interval=state.interval,
        )

    @app.get("/api/servers")
    async def list_servers():
        return {s.name: {"host": s.host, "port": s.port, "user": s.user} for s in state.servers}

    @app.get("/api/metrics")
    async def get_metrics():
        async with state.cache_lock:
            return {name: m.model_dump() for name, m in state.metrics_cache.items()}

    @app.get("/api/stream")
    async def stream():
        async def _event_generator():
            last_sent: float = 0
            while True:
                try:
                    await asyncio.wait_for(state._update_event.wait(), timeout=state.interval * 2)
                except asyncio.TimeoutError:
                    pass
                state._update_event.clear()
                async with state.cache_lock:
                    payload = {name: m.model_dump() for name, m in state.metrics_cache.items()}
                    newest = max((m.timestamp for m in state.metrics_cache.values()), default=0)
                if newest <= last_sent:
                    continue
                last_sent = newest
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            _event_generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/connect/{name}")
    async def connect_info(name: str):
        s = server_map.get(name)
        if not s:
            return JSONResponse({"error": f"Unknown server: {name}"}, status_code=404)
        ssh_cmd = f"ssh {s.user}@{s.host} -p {s.port}"
        vscode_url = f"vscode://vscode-remote/ssh-remote+{s.user}@{s.host}:{s.port}"
        return {"name": s.name, "ssh_cmd": ssh_cmd, "vscode_url": vscode_url,
                "host": s.host, "port": s.port, "user": s.user}

    @app.post("/api/terminal/{name}")
    async def open_terminal(name: str):
        s = server_map.get(name)
        if not s:
            return JSONResponse({"error": f"Unknown server: {name}"}, status_code=404)
        ssh_cmd = f"ssh {s.user}@{s.host} -p {s.port}"
        err = await asyncio.to_thread(_launch_terminal, ssh_cmd)
        if err:
            return JSONResponse({"error": err, "ssh_cmd": ssh_cmd}, status_code=500)
        return {"ok": True, "ssh_cmd": ssh_cmd}

    @app.post("/api/vscode/{name}")
    async def open_vscode(name: str):
        s = server_map.get(name)
        if not s:
            return JSONResponse({"error": f"Unknown server: {name}"}, status_code=404)
        ssh_cmd = f"ssh {s.user}@{s.host} -p {s.port}"
        err = await asyncio.to_thread(_launch_vscode, ssh_cmd, name)
        if err:
            return JSONResponse({"error": err, "ssh_cmd": ssh_cmd}, status_code=500)
        return {"ok": True, "ssh_cmd": ssh_cmd}

    return app


async def _probe_loop(state: AppState) -> None:
    logger.info("Probe loop started: %d servers", len(state.servers))
    while True:
        start = time.monotonic()
        try:
            new_metrics = await probe_all_servers(state.servers, timeout=state.ssh_timeout)
        except Exception:
            logger.exception("Probe cycle failed")
            await asyncio.sleep(state.interval)
            continue
        async with state.cache_lock:
            state.metrics_cache.update(new_metrics)
        state._update_event.set()
        elapsed = time.monotonic() - start
        await asyncio.sleep(max(0, state.interval - elapsed))


async def _my_usage_loop(state: AppState) -> None:
    """Periodically query per-process GPU stats for our tasks.
    Matches by conda env name 'pegasusmodye' — unique to our training.
    """
    import asyncssh

    logger.info("My GPU usage loop started (30s interval)")

    while True:
        await asyncio.sleep(30)
        try:
            from pathlib import Path as _P
            _here = _P(__file__).resolve().parent.parent
            if str(_here) not in sys.path:
                sys.path.insert(0, str(_here))
            from gpu_jobs.registry import load_registry

            reg = load_registry()
            n_jobs = len(reg.jobs)
            my_gpus: dict[str, set[int]] = {}
            for job in reg.jobs.values():
                for task in job.tasks:
                    if task.status == "running" and task.server:
                        my_gpus.setdefault(task.server, set()).add(task.gpu)

            logger.info("my-gpu-usage: %d jobs, %d servers with running tasks", n_jobs, len(my_gpus))

            if not my_gpus:
                async with state._my_usage_lock:
                    state.my_gpu_usage = {}
                continue

            host_map = {s.name: (s.host, s.port, s.identity_file) for s in state.servers}
            result: dict[str, dict[int, dict]] = {}

            for srv_name, gpu_indices in my_gpus.items():
                info = host_map.get(srv_name)
                if not info:
                    continue
                host, port, key_path = info
                try:
                    ck = {"known_hosts": None}
                    if key_path:
                        ck["client_keys"] = [key_path]
                    async with asyncssh.connect(host, port=port, username="user", **ck) as conn:
                        # Bus ID → GPU index
                        bus_r = await conn.run(
                            "nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null",
                            check=False,
                        )
                        bus_to_gpu: dict[str, int] = {}
                        for line in bus_r.stdout.strip().split("\n"):
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) >= 2:
                                try:
                                    bus_to_gpu[parts[1]] = int(parts[0])
                                except ValueError:
                                    pass

                        # Per-process stats
                        proc_r = await conn.run(
                            "nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_gpu_memory,process_name "
                            "--format=csv,noheader 2>/dev/null",
                            check=False,
                        )
                        our_mem: dict[int, int] = {}  # gpu → total_mem_mb
                        for line in proc_r.stdout.strip().split("\n"):
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) < 4:
                                continue
                            try:
                                bus = parts[1]
                                mem_str = parts[2].replace(" MiB", "")
                                mem_mb = int(mem_str)
                                proc_name = parts[3]
                                gpu_idx = bus_to_gpu.get(bus)
                            except (ValueError, KeyError):
                                continue
                            if gpu_idx is None:
                                continue
                            if _OUR_ENV_MARKER in proc_name:
                                our_mem[gpu_idx] = our_mem.get(gpu_idx, 0) + mem_mb

                        # Build result for this server's GPUs
                        for gpu_idx in gpu_indices:
                            total_util = 0.0
                            total_mem = 0.0
                            async with state.cache_lock:
                                cached = state.metrics_cache.get(srv_name)
                                if cached and not cached.error:
                                    for g in cached.gpu_info:
                                        if g.index == gpu_idx:
                                            total_util = g.utilization_gpu
                                            total_mem = g.memory_used_mb
                                            break

                            my_mem_val = our_mem.get(gpu_idx, 0)
                            my_util = (my_mem_val / total_mem * total_util) if total_mem > 0 else 0.0

                            result.setdefault(srv_name, {})[gpu_idx] = {
                                "my_util_pct": round(min(my_util, total_util), 1),
                                "my_mem_mb": round(my_mem_val),
                                "total_util_pct": round(total_util, 1),
                                "total_mem_mb": round(total_mem),
                            }
                except Exception:
                    continue

            async with state._my_usage_lock:
                state.my_gpu_usage = result
        except Exception:
            logger.exception("my-gpu-usage loop error")
