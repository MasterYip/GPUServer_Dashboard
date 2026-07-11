"""Rich-based terminal dashboard for GPU server monitoring."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import load_servers
from .models import ServerConfig, ServerMetrics
from .probe import probe_all_servers

logger = logging.getLogger(__name__)

_TERMINALS = [
    "gnome-terminal", "konsole", "xfce4-terminal", "mate-terminal",
    "lxterminal", "terminator", "tilix", "kitty", "alacritty",
    "wezterm", "xterm", "uxterm",
]
_TERMINAL_ARGS: dict[str, list[str]] = {
    "gnome-terminal":  ["--", "bash", "-c"],
    "konsole":         ["-e", "bash", "-c"],
    "xfce4-terminal":  ["-e", "bash", "-c"],
    "mate-terminal":   ["-e", "bash", "-c"],
    "lxterminal":      ["-e", "bash", "-c"],
    "terminator":      ["-e", "bash", "-c"],
    "tilix":           ["-e", "bash", "-c"],
    "kitty":           ["bash", "-c"],
    "alacritty":       ["-e", "bash", "-c"],
    "wezterm":         ["start", "--", "bash", "-c"],
    "xterm":           ["-e", "bash", "-c"],
    "uxterm":          ["-e", "bash", "-c"],
}


def _bar_color(pct: float) -> str:
    if pct < 60:   return "green"
    if pct < 85:   return "yellow"
    return "red"


def _status_dot(m: ServerMetrics | None, now: float) -> str:
    if m is None:      return "[dim]⬤[/dim]"
    if m.error:        return "[red]⬤[/red]"
    age = now - m.timestamp
    if age > 60:       return "[red]⬤[/red]"
    if age > 30:       return "[yellow]⬤[/yellow]"
    return "[green]⬤[/green]"


def _age_str(ts: float, now: float) -> str:
    s = max(0, int(now - ts))
    if s < 10:  return "just now"
    if s < 60:  return f"{s}s ago"
    return f"{s // 60}m ago"


def _bar(pct: float, width: int = 16) -> str:
    filled = int(round(pct / 100 * width))
    empty = width - filled
    color = _bar_color(pct)
    return f"[{color}]" + "█" * filled + "[dim]" + "░" * empty + "[/dim]"


def _launch_terminal(ssh_cmd: str) -> str | None:
    for name in _TERMINALS:
        path = shutil.which(name)
        if path:
            args = _TERMINAL_ARGS.get(name, ["-e", "bash", "-c"])
            full = f"{ssh_cmd} ; echo '---'; read -p 'Press Enter to close...'"
            try:
                subprocess.Popen(
                    [path] + args + [full],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
                return None
            except OSError:
                continue
    return "No terminal emulator found"


def build_dashboard(
    servers: list[ServerConfig],
    metrics_cache: dict[str, ServerMetrics],
) -> list:
    """Build dashboard renderables: header + server cards + footer."""
    now = time.time()

    # Header
    header = Table(show_header=False, expand=True, padding=0, box=None)
    online = sum(1 for s in servers if not _get_err(metrics_cache.get(s.name)))
    errored = sum(1 for s in servers if _get_err(metrics_cache.get(s.name)))
    h = Text()
    h.append("🖥️  GPU Server Dashboard", style="bold white")
    h.append(f"    {online} online", style="green")
    if errored:
        h.append(f"  {errored} error", style="red")
    h.append(f"    {datetime.now().strftime('%H:%M:%S')}", style="dim")
    header.add_row(h)

    # Cards in a 2-column grid
    cards = [_build_card(i, s, metrics_cache.get(s.name), now)
             for i, s in enumerate(servers)]

    grid = Table(show_header=False, expand=True, padding=(0, 1), box=None)
    grid.add_column(ratio=1, no_wrap=True)
    grid.add_column(ratio=1, no_wrap=True)
    for i in range(0, len(cards), 2):
        left = cards[i]
        right = cards[i + 1] if i + 1 < len(cards) else Panel("", border_style="dim")
        grid.add_row(left, right)

    # Footer
    footer = Table(show_header=False, expand=True, padding=0, box=None)
    f = Text()
    f.append("[bold]Keys:[/bold] ", style="dim")
    for i, s in enumerate(servers):
        f.append(f"{i}", style="bold cyan")
        f.append(f"={s.name}  ", style="dim")
    f.append("q", style="bold red")
    f.append(" Quit", style="dim")
    footer.add_row(f)

    return [header, Panel(grid, border_style="dim"), footer]


def _get_err(m: ServerMetrics | None) -> str | None:
    if m is None:
        return "probing..."
    return m.error


def _build_card(idx: int, server: ServerConfig, m: ServerMetrics | None,
                now: float) -> Panel:
    """Build a single server card panel."""
    dot = _status_dot(m, now)
    tag = f"[bold cyan]({idx})[/bold cyan]" if idx < 10 else f"[dim]({idx})[/dim]"
    title = f"{tag} {dot} [bold]{server.name}[/bold]"

    if m is None:
        return Panel(Text("  [dim]probing...[/dim]"), title=title,
                      title_align="left", border_style="dim", padding=(1, 2))

    if m.error:
        return Panel(Text(f"  [red]{m.error}[/red]"), title=title,
                      title_align="left", border_style="red", padding=(1, 2))

    lines: list[str] = []
    gpus = m.gpu_info

    if gpus:
        for g in gpus:
            mem_pct = (g.memory_used_mb / g.memory_total_mb * 100) if g.memory_total_mb > 0 else 0
            mem_used = g.memory_used_mb / 1024
            mem_total = g.memory_total_mb / 1024

            extras: list[str] = []
            if g.temperature_gpu is not None:
                extras.append(f"{g.temperature_gpu:.0f}°C")
            if g.power_draw_w is not None:
                extras.append(f"{g.power_draw_w:.0f}W")
            extra = f"  [dim]{' · '.join(extras)}[/dim]" if extras else ""

            lines.append(f" [bold dim]GPU{g.index}[/] {g.name}{extra}")
            lines.append(f"  GPU {_bar(g.utilization_gpu)} {g.utilization_gpu:5.1f}%")
            lines.append(f"  MEM {_bar(mem_pct)} {mem_pct:5.1f}%  {mem_used:.0f}/{mem_total:.0f}G")
    else:
        lines.append(" [dim]no GPU[/dim]")

    lines.append(f" CPU  {_bar(m.cpu_percent)} {m.cpu_percent:5.1f}%")
    lines.append(f" RAM  {_bar(m.ram_percent)} {m.ram_percent:5.1f}%  "
                 f"{m.ram_used_gb:.0f}/{m.ram_total_gb:.0f}G")
    lines.append(f" [dim]⏱ {_age_str(m.timestamp, now)}[/dim]")

    body = Text("\n".join(lines), no_wrap=True)
    max_pct = max([m.cpu_percent, m.ram_percent] +
                  [g.utilization_gpu for g in gpus] + [0])
    border = _bar_color(max_pct)
    return Panel(body, title=title, title_align="left",
                  border_style=border, padding=(1, 2))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_tui(
    config_path: str,
    interval: int = 10,
    ssh_timeout: int = 5,
) -> None:
    """Run the interactive terminal dashboard.  Press Q to exit."""
    servers = load_servers(config_path)
    if not servers:
        print("No servers found in config.")
        return

    console = Console()
    metrics_cache: dict[str, ServerMetrics] = {}

    async def _probe_once():
        try:
            new = await probe_all_servers(servers, timeout=ssh_timeout)
        except Exception:
            logger.exception("Probe cycle failed")
            return
        metrics_cache.update(new)

    console.print("[dim]Connecting to servers...[/dim]")
    await _probe_once()

    def _render():
        rendered = build_dashboard(servers, metrics_cache)
        # Flatten into single renderable via Table
        t = Table.grid()
        for r in rendered:
            t.add_row(r)
        return t

    # Stdin reader for keyboard shortcuts
    async def _key_reader():
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while True:
            try:
                ch = await asyncio.wait_for(reader.read(1), timeout=0.3)
            except asyncio.TimeoutError:
                continue
            if not ch:
                continue
            key = ch.decode(errors="replace")
            # Digit 0-9 → launch terminal for that server
            if key.isdigit():
                idx = int(key)
                if 0 <= idx < len(servers):
                    s = servers[idx]
                    ssh_cmd = f"ssh {s.user}@{s.host} -p {s.port}"
                    await asyncio.to_thread(_launch_terminal, ssh_cmd)
            elif key.lower() == "q":
                break

    key_task = asyncio.create_task(_key_reader())

    with Live(_render(), console=console, refresh_per_second=4, screen=True) as live:
        while not key_task.done():
            await asyncio.sleep(interval)
            await _probe_once()
            live.update(_render())

    key_task.cancel()
    try:
        await key_task
    except asyncio.CancelledError:
        pass
