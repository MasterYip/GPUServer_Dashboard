"""Rich-based terminal dashboard for GPU server monitoring.

Vim-style full-screen dashboard.  j/k or arrows to scroll, 0-9 for terminal, q to quit.
"""

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

# Cached terminal size
_term_w, _term_h = 80, 24

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar_color(pct: float) -> str:
    if pct < 60:   return "green"
    if pct < 85:   return "yellow"
    return "red"


def _status_dot(m: ServerMetrics | None, now: float) -> str:
    if m is None:      return "[dim]●[/dim]"
    if m.error:        return "[red]●[/red]"
    age = now - m.timestamp
    if age > 60:       return "[red]●[/red]"
    if age > 30:       return "[yellow]●[/yellow]"
    return "[green]●[/green]"


def _age_str(ts: float, now: float) -> str:
    s = max(0, int(now - ts))
    if s < 10:  return "just now"
    if s < 60:  return f"{s}s ago"
    return f"{s // 60}m ago"


def _bar(pct: float, width: int = 10) -> str:
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


# ---------------------------------------------------------------------------
# Card builder (compact — 1 bar per GPU with % + mem on same line)
# ---------------------------------------------------------------------------

def _build_card(idx: int, server: ServerConfig, m: ServerMetrics | None,
                now: float, width: int) -> Panel:
    """Build a single server card. Width-aware for grid layout."""
    dot = _status_dot(m, now)
    tag = f"[bold cyan]{idx}[/bold cyan]" if idx < 10 else f"[dim]{idx}[/dim]"
    title = Text.from_markup(f"{tag} {dot} [bold white]{server.name}[/bold white]")

    if m is None:
        return Panel(Text.from_markup("[dim]probing...[/dim]"),
                      title=title, title_align="left", border_style="dim",
                      padding=(1, 2))

    if m.error:
        return Panel(Text.from_markup(f"[red]{m.error}[/red]"),
                      title=title, title_align="left", border_style="red",
                      padding=(1, 2))

    # Adapt bar width to card width
    bar_w = max(6, min(16, (width - 50) // 2))

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
            extra = f" [dim]{'·'.join(extras)}[/dim]" if extras else ""

            lines.append(
                f" [bold dim]G{g.index}[/] {g.name}{extra}"
            )
            lines.append(
                f"  GPU [{_bar_color(g.utilization_gpu)}]{_bar(g.utilization_gpu, bar_w)}[/]"
                f" {g.utilization_gpu:5.1f}%  "
                f"MEM [{_bar_color(mem_pct)}]{_bar(mem_pct, bar_w)}[/] {mem_pct:5.1f}%  "
                f"{mem_used:.0f}/{mem_total:.0f}G"
            )
    else:
        lines.append(" [dim]no GPU[/dim]")

    lines.append(
        f" CPU  [{_bar_color(m.cpu_percent)}]{_bar(m.cpu_percent, bar_w)}[/]"
        f" {m.cpu_percent:5.1f}%  "
        f"RAM  [{_bar_color(m.ram_percent)}]{_bar(m.ram_percent, bar_w)}[/]"
        f" {m.ram_percent:5.1f}%  {m.ram_used_gb:.0f}/{m.ram_total_gb:.0f}G"
    )
    lines.append(f" [dim]⏱ {_age_str(m.timestamp, now)}[/dim]")

    body_str = "\n".join(lines)
    max_pct = max([m.cpu_percent, m.ram_percent] +
                  [g.utilization_gpu for g in gpus] + [0])
    return Panel(body_str, title=title, title_align="left",
                  border_style=_bar_color(max_pct), padding=(1, 2))


def _detect_term_size(console: Console) -> tuple[int, int]:
    """Get terminal width/height, with fallback."""
    try:
        sz = shutil.get_terminal_size()
        return sz.columns, sz.lines
    except Exception:
        return 80, 24


def _build_dashboard(
    servers: list[ServerConfig],
    metrics_cache: dict[str, ServerMetrics],
    scroll_offset: int,
    console: Console,
) -> Table:
    """Build the dashboard with scroll-aware card grid."""
    now = time.time()
    global _term_w, _term_h
    _term_w, _term_h = _detect_term_size(console)

    # Choose columns: 2 if wide enough, else 1
    ncols = 2 if _term_w >= 100 else 1
    card_width = _term_w // ncols - (4 if ncols == 2 else 0)

    # Build all cards
    cards = [_build_card(i, s, metrics_cache.get(s.name), now, card_width)
             for i, s in enumerate(servers)]

    # Estimate card heights (conservative guess based on GPU count + CPU/RAM lines)
    def _card_height(s: ServerConfig) -> int:
        m = metrics_cache.get(s.name)
        gpu_count = len(m.gpu_info) if (m and not m.error) else 0
        # Each GPU: 2 lines (header + bars), CPU/RAM: 1 line, age: 1 line
        # Panel border adds ~3 lines, padding adds 2
        gpu_lines = gpu_count * 2 if gpu_count else 1  # "no GPU" = 1 line
        return gpu_lines + 1 + 4  # base lines + CPU/RAM line + border/padding

    # Calculate visible area
    header_h = 2
    footer_h = 2
    body_h = _term_h - header_h - footer_h

    # Compute row heights per column layout
    if ncols == 2:
        # Pair cards: each "visual row" height = max(left card height, right card height)
        rows: list[int] = []
        for i in range(0, len(cards), 2):
            left_h = _card_height(servers[i])
            right_h = _card_height(servers[i + 1]) if i + 1 < len(servers) else 0
            rows.append(max(left_h, right_h))
    else:
        rows = [_card_height(s) for s in servers]

    # Clamp scroll offset
    total_rows = len(rows)
    visible_row_slots = max(1, body_h // 6)  # ~6 lines per row (rough)
    max_offset = max(0, total_rows - visible_row_slots)
    scroll_offset = max(0, min(scroll_offset, max_offset))

    # Determine which rows to render
    visible_row_indices: set[int] = set()
    cur_h = 0
    for ri in range(scroll_offset, total_rows):
        if cur_h + rows[ri] > body_h:
            break
        visible_row_indices.add(ri)
        cur_h += rows[ri]

    # ---- Assemble output ----
    outer = Table.grid(padding=0)
    outer.add_column(ratio=1)

    # Header
    online = sum(1 for s in servers if not _get_err(metrics_cache.get(s.name)))
    errored = sum(1 for s in servers if _get_err(metrics_cache.get(s.name)))
    h = Text.from_markup(
        f"[bold white]🖥️  GPU Server Dashboard[/bold white]"
        f"    [green]{online} online[/green]"
        + (f"  [red]{errored} error[/red]" if errored else "")
        + f"    [dim]{datetime.now().strftime('%H:%M:%S')}[/dim]"
        + (f"  [dim]▼{scroll_offset}/{max_offset}[/dim]" if max_offset > 0 else "")
    )
    header_table = Table.grid(padding=0)
    header_table.add_column(ratio=1)
    header_table.add_row(h)
    outer.add_row(header_table)

    # Card rows (2-column grid)
    grid = Table.grid(padding=(0, 1))
    if ncols == 2:
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

    for ri in range(scroll_offset, total_rows):
        if ri not in visible_row_indices:
            continue
        if ncols == 2:
            left_idx = ri * 2
            right_idx = left_idx + 1
            left = cards[left_idx]
            right = cards[right_idx] if right_idx < len(cards) else ""
            grid.add_row(left, right)
        else:
            grid.add_row(cards[ri])

    outer.add_row(grid)

    # Fill remaining space
    remaining = body_h - cur_h
    if remaining > 0:
        outer.add_row(Text(" " * remaining))

    # Footer
    scrolled = len(visible_row_indices) < total_rows
    footer_text = Text.from_markup(
        "[dim][bold]j/k ↓↑[/bold] scroll[/dim]"
        + f"  [dim][bold]0-9[/bold] terminal[/dim]"
        + f"  [dim][bold]q[/bold] quit[/dim]"
        + (f"  [yellow]{scroll_offset + 1}-{scroll_offset + len(visible_row_indices)}/"
           f"{total_rows}[/yellow]" if scrolled else "")
    )
    footer_table = Table.grid(padding=0)
    footer_table.add_column(ratio=1)
    footer_table.add_row(footer_text)
    outer.add_row(footer_table)

    return outer


def _get_err(m: ServerMetrics | None) -> str | None:
    if m is None:
        return "probing..."
    return m.error


# ---------------------------------------------------------------------------
# Keyboard input — buffered for escape sequences (arrow keys etc.)
# ---------------------------------------------------------------------------

_ARROW_MAP: dict[str, str] = {
    "A": "up", "B": "down", "C": "right", "D": "left",
    "5~": "pgup", "6~": "pgdn",
}

_ESCAPE_TIMEOUT = 0.05  # seconds to wait after ESC for sequence completion


async def _read_keys(queue: asyncio.Queue[str]) -> None:
    """Read stdin bytes and enqueue logical key names ('up','down','pgup','pgdn','q','0'-'9', etc.)."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    buf = b""

    while True:
        ch = await reader.read(1)
        if not ch:
            break

        # ESC starts a sequence
        if ch == b"\x1b":
            try:
                buf = ch
                nxt = await asyncio.wait_for(reader.read(1), timeout=_ESCAPE_TIMEOUT)
                buf += nxt
                if nxt == b"[":
                    rest = b""
                    while True:
                        n = await asyncio.wait_for(reader.read(1), timeout=_ESCAPE_TIMEOUT)
                        rest += n
                        if n in b"A B C D H F" or rest[-2:] in (b"5~", b"6~"):
                            break
                    key = _ARROW_MAP.get(rest.decode(errors="replace"), "")
                    if key:
                        await queue.put(key)
                else:
                    await queue.put("escape")
            except asyncio.TimeoutError:
                await queue.put("escape")
            continue

        # Regular character
        key = ch.decode(errors="replace")
        await queue.put(key)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_tui(
    config_path: str,
    interval: int = 10,
    ssh_timeout: int = 5,
) -> None:
    """Run the interactive terminal dashboard."""
    servers = load_servers(config_path)
    if not servers:
        print("No servers found in config.")
        return

    console = Console()
    metrics_cache: dict[str, ServerMetrics] = {}
    scroll_offset = 0

    async def _probe_once():
        try:
            new = await probe_all_servers(servers, timeout=ssh_timeout)
        except Exception:
            logger.exception("Probe cycle failed")
            return
        metrics_cache.update(new)

    console.clear()
    await _probe_once()

    # Start the key reader background task
    queue: asyncio.Queue[str] = asyncio.Queue()
    key_task = asyncio.create_task(_read_keys(queue))

    # Track probe timing separately from key handling
    last_probe = time.monotonic()

    def _render():
        return _build_dashboard(servers, metrics_cache, scroll_offset, console)

    async def _handle_key(key: str) -> str | None:
        nonlocal scroll_offset
        if key == "q":
            return "quit"
        elif key in ("j", "down"):
            scroll_offset += 1
        elif key in ("k", "up"):
            scroll_offset = max(0, scroll_offset - 1)
        elif key == "pgdn":
            scroll_offset += 5
        elif key == "pgup":
            scroll_offset = max(0, scroll_offset - 5)
        elif key == "g":
            scroll_offset = 0
        elif key == "G":
            scroll_offset = 999
        elif key.isdigit():
            idx = int(key)
            if 0 <= idx < len(servers):
                s = servers[idx]
                ssh_cmd = f"ssh {s.user}@{s.host} -p {s.port}"
                await asyncio.to_thread(_launch_terminal, ssh_cmd)
        return None

    with Live(_render(), console=console, refresh_per_second=60,
              screen=True) as live:

        while True:
            # Poll keys at high frequency — drain all pending immediately
            try:
                key = await asyncio.wait_for(queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                key = None

            # Process all queued keys
            while key is not None:
                result = await _handle_key(key)
                if result == "quit":
                    key_task.cancel()
                    return
                # Try to get next key without blocking
                try:
                    key = queue.get_nowait()
                except asyncio.QueueEmpty:
                    key = None

            # Re-render immediately for scroll changes
            live.update(_render())

            # Probe only if interval has elapsed (or no probe running)
            now = time.monotonic()
            if now - last_probe >= interval:
                last_probe = now
                await _probe_once()
                live.update(_render())

    key_task.cancel()
    try:
        await key_task
    except asyncio.CancelledError:
        pass
