#!/usr/bin/env python3
"""
Storage bandwidth benchmark across GPU servers via SSH.

Tests read & write speeds on key mount points (local SSD, /data, NAS, NFS)
and produces a ranked report.

Usage:
  # Test all servers
  python ./scripts/benchmark_storage.py --config servers.yaml

  # Test only 4090 servers
  python ./scripts/benchmark_storage.py --config servers_rp.yaml --size 2048 --filter 4090

  # Quick test (smaller file)
  python ./scripts/benchmark_storage.py --config servers_rp.yaml --size 512 --filter 4090

  # Save report to file
  python ./scripts/benchmark_storage.py --config servers.yaml --filter 4090 -o report.txt
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add dashboard to path so we can reuse probe infrastructure
_SCRIPT_DIR = Path(__file__).resolve().parent
_DASHBOARD_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_DASHBOARD_DIR))

from dashboard.config import load_servers
from dashboard.models import ServerConfig

# ---------------------------------------------------------------------------
# Benchmark logic (runs on remote via SSH)
# ---------------------------------------------------------------------------

# Mount points to test on each server.  We auto-detect which exist.
CANDIDATE_MOUNTS: list[tuple[str, str]] = [
    ("/",            "local-SSD-root"),
    ("/data",        "local-data-vol"),
    ("/mnt/nas",     "NAS-shared"),
    # Per-server NFS mounts are discovered dynamically via df
]

def _build_bench_script(size_mb: int, count: int = 1) -> str:
    """Build the benchmark bash script using simple placeholder substitution.

    Uses sudo for file operations since mount points are typically
    root-owned.  Falls back to non-sudo if the user is root or
    sudo is not available.
    """
    script = r'''#!/usr/bin/env bash
set -uo pipefail
SIZE_MB=_SIZE_MB_
TESTFILE=".bw_$$.tmp"

# Determine whether to use sudo for benchmark writes
SUDO=""
if [[ $EUID -ne 0 ]] && command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
    SUDO="sudo"
fi

echo "=== STORAGE BENCHMARK ==="
echo "Host: $(hostname)  Test size: ${SIZE_MB}M"

CANDIDATES="/ /data /mnt/nas"
EXTRA=$(df 2>/dev/null | awk 'NR>1 && $NF ~ /^[/]mnt[/]/ {print $NF}' | sort -u 2>/dev/null || true)
ALL_MOUNTS=$(printf '%s\n' $CANDIDATES $EXTRA | sort -u)

echo "DISCOVERED: $ALL_MOUNTS"

for MOUNT in $ALL_MOUNTS; do
    [[ -d "$MOUNT" ]] || { echo "SKIP mount=$MOUNT reason=not_a_directory"; continue; }
    TESTPATH="$MOUNT/$TESTFILE"
    $SUDO rm -f "$TESTPATH" 2>/dev/null || true

    # Quick write test
    if ! $SUDO touch "$TESTPATH" 2>/dev/null; then
        echo "SKIP mount=$MOUNT reason=touch_failed"
        continue
    fi
    $SUDO rm -f "$TESTPATH" 2>/dev/null || true

    # ---- WRITE benchmark ----
    # Write SIZE_MB megabytes in one dd operation
    START=$(date +%s%N)
    $SUDO dd if=/dev/zero of="$TESTPATH" bs=1M count=$SIZE_MB 2>/dev/null
    RC=$?
    END=$(date +%s%N)
    ELAPSED_NS=$((END - START))

    BYTES_WRITTEN=$(stat -c%s "$TESTPATH" 2>/dev/null || echo 0)
    if [[ $RC -ne 0 || $BYTES_WRITTEN -le 0 ]]; then
        echo "SKIP mount=$MOUNT reason=dd_write_failed_(rc=$RC_bytes=$BYTES_WRITTEN)"
        $SUDO rm -f "$TESTPATH" 2>/dev/null || true
        continue
    fi

    if [[ $ELAPSED_NS -gt 0 ]]; then
        ELAPSED_S=$(awk "BEGIN {printf \"%.3f\", $ELAPSED_NS / 1000000000}")
        WRITE_MBS=$(awk "BEGIN {printf \"%.1f\", ($BYTES_WRITTEN / 1048576) / $ELAPSED_S}")
    else
        WRITE_MBS="0"
    fi

    # ---- READ benchmark ----
    sync 2>/dev/null || true
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    START=$(date +%s%N)
    $SUDO dd if="$TESTPATH" of=/dev/null bs=1M count=$SIZE_MB 2>/dev/null || true
    END=$(date +%s%N)
    ELAPSED_NS=$((END - START))
    if [[ $ELAPSED_NS -gt 0 ]]; then
        ELAPSED_S=$(awk "BEGIN {printf \"%.3f\", $ELAPSED_NS / 1000000000}")
        READ_MBS=$(awk "BEGIN {printf \"%.1f\", ($BYTES_WRITTEN / 1048576) / $ELAPSED_S}")
    else
        READ_MBS="0"
    fi

    # Cleanup
    $SUDO rm -f "$TESTPATH" 2>/dev/null || true

    FSTYPE=$(df -T "$MOUNT" 2>/dev/null | tail -1 | awk '{print $2}' || echo "unknown")

    # Detect disk type: SSD vs HDD vs remote vs ram
    DISK_TYPE="unknown"
    case "$FSTYPE" in
        nfs|nfs4|cifs|smb|fuse.*|glusterfs|cephfs|lustre)
            DISK_TYPE="remote" ;;
        tmpfs|ramfs|devtmpfs)
            DISK_TYPE="ram" ;;
        *)
            # Resolve the underlying block device
            DEV=$(df "$MOUNT" 2>/dev/null | awk 'NR>1 {print $1}' | sed 's|/dev/||')
            # Strip partition numbers: sda1→sda, nvme0n1p2→nvme0n1
            BASE_DEV=$(echo "$DEV" | sed -E 's/p?[0-9]+$//')
            if [[ -n "$BASE_DEV" && -f "/sys/block/$BASE_DEV/queue/rotational" ]]; then
                ROT=$(cat "/sys/block/$BASE_DEV/queue/rotational" 2>/dev/null)
                if [[ "$ROT" == "0" ]]; then
                    DISK_TYPE="ssd"
                elif [[ "$ROT" == "1" ]]; then
                    DISK_TYPE="hdd"
                fi
            fi
            ;;
    esac

    echo "RESULT mount=$MOUNT fstype=$FSTYPE disk_type=$DISK_TYPE write_mbs=$WRITE_MBS read_mbs=$READ_MBS"
done

# Print cache caveat for network filesystems
HAS_NFS=$(df -T 2>/dev/null | awk 'NR>1 && ($2=="nfs" || $2=="nfs4") {print "1"; exit}')
if [[ -n "$HAS_NFS" ]]; then
    echo ""
    echo "NOTE: 'drop_caches' only clears the local client cache, not the NFS server's"
    echo "page cache. Since the file was just written, the NFS server still has it in"
    echo "RAM, so read speeds for remote mounts may be inflated (server cache, not disk)."
    echo "To get real NFS read speeds, drop caches on the NFS server or read a cold file."
fi

echo "=== DONE ==="
'''
    return script.replace("_SIZE_MB_", str(size_mb))


@dataclass
class MountResult:
    mount: str
    fstype: str
    disk_type: str
    write_mbs: float
    read_mbs: float


@dataclass
class ServerResult:
    server_name: str
    error: str | None = None
    mount_results: list[MountResult] = field(default_factory=list)
    skipped_mounts: list[str] = field(default_factory=list)
    raw_mounts: str = ""


# ---------------------------------------------------------------------------
# SSH runner (uses asyncssh like dashboard, but simpler sync interface okay)
# ---------------------------------------------------------------------------

async def _run_bench_on_server(
    server: ServerConfig,
    size_mb: int,
    count: int,
    timeout: int = 30,
) -> ServerResult:
    """SSH into one server, run the benchmark script, parse results."""
    import asyncssh

    script = _build_bench_script(size_mb, count)

    try:
        conn_kwargs: dict = {}
        identity_file = server.identity_file or "~/.ssh/id_rsa"
        conn_kwargs["client_keys"] = [os.path.expanduser(identity_file)]
        conn_kwargs["known_hosts"] = None

        async with asyncssh.connect(
            server.host, port=server.port,
            username=server.user, **conn_kwargs,
        ) as conn:
            result = await asyncio.wait_for(
                conn.run(f"bash -s << 'BENCHMARK_EOF'\n{script}\nBENCHMARK_EOF", check=False),
                timeout=timeout,
            )
    except asyncio.TimeoutError:
        return ServerResult(server_name=server.name, error=f"Timeout ({timeout}s)")
    except Exception as exc:
        return ServerResult(server_name=server.name, error=str(exc))

    output = result.stdout or ""
    stderr = result.stderr or ""

    sr = ServerResult(server_name=server.name)
    if result.exit_status != 0:
        # Show stderr + partial stdout for debugging
        debug = (stderr + "\n" + output).strip()[:500]
        sr.error = f"Exit {result.exit_status}: {debug}"
        return sr

    # Parse output
    sr.raw_mounts = output[:2000]  # keep full output for inspection
    for line in output.splitlines():
        line = line.strip()

        if line.startswith("SKIP "):
            # Format: SKIP mount=/data reason=not_writable
            try:
                parts = dict(
                    item.split("=", 1) for item in line.removeprefix("SKIP ").split()
                )
                mount = parts.get("mount", "?")
                reason = parts.get("reason", "?")
                sr.skipped_mounts.append(f"{mount} ({reason})")
            except (ValueError, KeyError):
                sr.skipped_mounts.append(line)

        if line.startswith("RESULT "):
            parts = dict(
                item.split("=", 1) for item in line.removeprefix("RESULT ").split()
            )
            try:
                sr.mount_results.append(MountResult(
                    mount=parts["mount"],
                    fstype=parts.get("fstype", "unknown"),
                    disk_type=parts.get("disk_type", "unknown"),
                    write_mbs=float(parts.get("write_mbs", 0)),
                    read_mbs=float(parts.get("read_mbs", 0)),
                ))
            except (ValueError, KeyError):
                continue

    return sr


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def _fmt(val: float) -> str:
    if val <= 0:
        return "     --"
    if val < 10:
        return f" {val:6.1f}"
    if val < 100:
        return f" {val:6.1f}"
    return f"{val:7.1f}"


def _grade(val: float) -> str:
    if val <= 0: return "?"
    if val < 50: return "\033[31mSLOW\033[0m"
    if val < 200: return "\033[33mOK\033[0m"
    if val < 500: return "\033[32mFAST\033[0m"
    return "\033[32m\033[1mBLAZING\033[0m"


def print_report(results: list[ServerResult], size_mb: int) -> None:
    """Print a formatted bandwidth report using Rich tables."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Collect all unique mount labels with disk type
    DISK_TYPE_ICONS: dict[str, str] = {
        "ssd":     "⚡",
        "hdd":     "🔄",
        "remote":  "🌐",
        "ram":     "🧠",
        "unknown": "❓",
    }
    all_mounts: dict[str, str] = {}
    for sr in results:
        for mr in sr.mount_results:
            icon = DISK_TYPE_ICONS.get(mr.disk_type, "❓")
            all_mounts[mr.mount] = f"{mr.mount} {icon}{mr.disk_type} ({mr.fstype})"
    mount_order = sorted(all_mounts.keys())

    # ---- Main results table ------------------------------------------------
    main = Table(
        title=f"Storage Bandwidth Report — {size_mb}M test file",
        title_style="bold white",
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
    )
    main.add_column("Server", style="bold", width=14)
    for mp in mount_order:
        main.add_column(f"W {all_mounts[mp]}", justify="right", width=10)
        main.add_column(f"R {all_mounts[mp]}", justify="right", width=10)

    for sr in sorted(results, key=lambda r: r.server_name):
        if sr.error and not sr.mount_results:
            row = [f"[red]{sr.server_name}[/red]", *["—"] * (len(mount_order) * 2)]
            main.add_row(*row, end_section=True)
            main.add_row("", *([f"[dim]{sr.error[:60]}[/dim]"] + [""] * (len(mount_order) * 2 - 1)))
            continue

        cells = [sr.server_name]
        for mp in mount_order:
            found = next((m for m in sr.mount_results if m.mount == mp), None)
            if found:
                wc = _bar_color_for_rich(found.write_mbs)
                rc = _bar_color_for_rich(found.read_mbs)
                cells.append(f"[{wc}]{found.write_mbs:.1f}[/{wc}]")
                cells.append(f"[{rc}]{found.read_mbs:.1f}[/{rc}]")
            else:
                cells.append("[dim]—[/dim]")
                cells.append("[dim]—[/dim]")
        main.add_row(*cells)

        # Skipped mounts
        if sr.skipped_mounts:
            for sm in sr.skipped_mounts:
                main.add_row("", *([f"[dim](skipped: {sm})[/dim]"] + [""] * (len(mount_order) * 2 - 1)), style="dim")

    console.print(main)

    # ---- Disk type legend --------------------------------------------------
    from rich.text import Text
    legend = Text.assemble(
        ("Disk types:  ", "dim"),
        ("⚡ SSD  ", "green"),
        ("🔄 HDD  ", "yellow"),
        ("🌐 remote (NFS/CIFS/…)  ", "cyan"),
        ("🧠 RAM (tmpfs)  ", "magenta"),
        ("❓ unknown", "dim"),
    )
    console.print(legend)
    console.print()

    # ---- Per-mount summary ------------------------------------------------
    summary = Table(
        title="Per-Mount Best Results",
        title_style="bold white",
        header_style="bold cyan",
        padding=(0, 1),
    )
    summary.add_column("Mount", style="bold")
    summary.add_column("Best Write", justify="right")
    summary.add_column("Server", justify="right")
    summary.add_column("Best Read", justify="right")
    summary.add_column("Server", justify="right")

    for mp in mount_order:
        bw, bs = "", ""
        br, rs = "", ""
        bw_val, br_val = 0.0, 0.0
        for sr in results:
            for mr in sr.mount_results:
                if mr.mount == mp:
                    if mr.write_mbs > bw_val:
                        bw_val = mr.write_mbs
                        bw, bs = f"[green]{bw_val:.1f} MB/s[/green]", sr.server_name
                    if mr.read_mbs > br_val:
                        br_val = mr.read_mbs
                        br, rs = f"[green]{br_val:.1f} MB/s[/green]", sr.server_name
        summary.add_row(all_mounts.get(mp, mp), bw or "—", bs or "—", br or "—", rs or "—")

    console.print(summary)

    # ---- Server ranking ------------------------------------------------
    rank_t = Table(
        title="Server Ranking (avg across mounts)",
        title_style="bold white",
        header_style="bold cyan",
        padding=(0, 1),
    )
    rank_t.add_column("#", justify="right", style="dim")
    rank_t.add_column("Server", style="bold")
    rank_t.add_column("Avg Write", justify="right")
    rank_t.add_column("Avg Read", justify="right")

    rankings: list[tuple[str, float, float]] = []
    for sr in results:
        if not sr.mount_results:
            continue
        avg_w = sum(m.write_mbs for m in sr.mount_results) / len(sr.mount_results)
        avg_r = sum(m.read_mbs for m in sr.mount_results) / len(sr.mount_results)
        rankings.append((sr.server_name, avg_w, avg_r))
    rankings.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, avg_w, avg_r) in enumerate(rankings, 1):
        wc = _bar_color_for_rich(avg_w)
        rc = _bar_color_for_rich(avg_r)
        rank_t.add_row(
            str(rank),
            name,
            f"[{wc}]{avg_w:.1f} MB/s[/{wc}]",
            f"[{rc}]{avg_r:.1f} MB/s[/{rc}]",
        )

    console.print(rank_t)

    # ---- Cache caveat ----------------------------------------------------
    from rich.panel import Panel
    console.print()
    console.print(Panel(
        "[yellow]⚠️  [bold]NFS read speeds may be inflated[/bold]\n\n"
        "[dim]'drop_caches' only clears the [italic]local[/italic] client cache, not the NFS server's "
        "page cache. Since the file was just written, the NFS server still has it in RAM, "
        "so read speeds for remote mounts ([cyan]nfs/nfs4[/cyan]) reflect server cache reads, "
        "not disk reads.\n\n"
        "To get real NFS read speeds: drop caches on the NFS server, or read a file "
        "that hasn't been recently accessed.[/dim]",
        title="[bold yellow]NFS Cache Caveat[/bold yellow]",
        border_style="yellow",
        padding=(0, 1),
    ))


def _bar_color_for_rich(val: float) -> str:
    if val <= 0:    return "dim"
    if val < 50:    return "red"
    if val < 200:   return "yellow"
    if val < 500:   return "green"
    return "bold green"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Storage bandwidth benchmark across GPU servers",
    )
    parser.add_argument(
        "--config", default="servers.yaml",
        help="Path to servers.yaml",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Only test servers matching this substring (e.g. '4090')",
    )
    parser.add_argument(
        "--size", type=int, default=1024,
        help="Test file size in MB (default: 1024)",
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of test blocks (increases accuracy but takes longer; default: 1)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="SSH timeout per server in seconds",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save report to file (also prints to stdout)",
    )
    args = parser.parse_args()

    servers = load_servers(args.config)
    if args.filter:
        servers = [s for s in servers if args.filter.lower() in s.name.lower()]

    if not servers:
        print(f"No servers found (filter='{args.filter}').")
        sys.exit(1)

    print(f"Benchmarking {len(servers)} server(s) with {args.size}M test file...")
    print("  (This may take a moment — running via SSH in parallel)\n")

    tasks = [
        _run_bench_on_server(s, size_mb=args.size, count=args.count, timeout=args.timeout)
        for s in servers
    ]
    results = await asyncio.gather(*tasks)

    # Print errors
    for sr in results:
        if sr.error:
            print(f"  \033[31m{sr.server_name}: {sr.error}\033[0m")

    # Print report
    print_report(results, args.size)

    # Save to file
    if args.output:
        import io

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_report(results, args.size)
        finally:
            sys.stdout = old_stdout

        with open(args.output, "w") as f:
            f.write(buf.getvalue())
        print(f"\nReport saved to: {args.output}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
