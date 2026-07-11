#!/usr/bin/env python3
"""
Storage bandwidth benchmark across GPU servers via SSH.

Tests read & write speeds on key mount points (local SSD, /data, NAS, NFS)
and produces a ranked report.

Usage:
  # Test all servers
  ./scripts/benchmark_storage.py --config servers.yaml

  # Test only 4090 servers
  ./scripts/benchmark_storage.py --config servers_rp.yaml --filter 4090

  # Quick test (smaller file)
  ./scripts/benchmark_storage.py --config servers_rp.yaml --size 512 --filter 4090

  # Save report to file
  ./scripts/benchmark_storage.py --config servers.yaml --filter 4090 -o report.txt
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

_BENCH_TEMPLATE = r'''
# Auto-generated storage benchmark — runs on remote server
set -euo pipefail
SIZE_MB=__SIZE_MB__
COUNT=__COUNT__
TESTFILE=".bandwidth_test_$$.tmp"

echo "=== STORAGE BENCHMARK ==="
echo "Host: $(hostname)"

# Discover all mount points
echo ""
echo "--- Mounts ---"
df -h --output=target,size,avail,fstype 2>/dev/null | tail -n +2 || \
df -h | awk '{{print $$NF, $$2, $$4, $$1}}'

echo ""
echo "--- Bandwidth Tests (file_size=$${{SIZE_MB}}M) ---"

# Target mount points hardcoded + dynamically discovered NFS mounts
for MOUNT in / /data /mnt/nas $$(df -h --output=target 2>/dev/null | grep '^/mnt/' || true); do
    [[ -d "$$MOUNT" ]] || continue
    [[ -w "$$MOUNT" ]] || continue

    TESTPATH="$$MOUNT/$$TESTFILE"

    # ---- WRITE ----
    rm -f "$$TESTPATH" 2>/dev/null || true
    START=$$(date +%s%N)
    dd if=/dev/zero of="$$TESTPATH" bs=1M count=$$COUNT oflag=direct conv=fdatasync 2>/tmp/dd_err.$$ || true
    END=$$(date +%s%N)
    ELAPSED_NS=$$((END - START))
    if [[ $$ELAPSED_NS -gt 0 ]]; then
        ELAPSED_S=$$(awk "BEGIN {{printf \"%.3f\", $$ELAPSED_NS / 1000000000}}")
        WRITE_MBS=$$(awk "BEGIN {{printf \"%.1f\", ($$SIZE_MB) / $$ELAPSED_S}}")
    else
        WRITE_MBS="0"
    fi

    # ---- READ ----
    if [[ -f "$$TESTPATH" ]]; then
        START=$$(date +%s%N)
        dd if="$$TESTPATH" of=/dev/null bs=1M count=$$COUNT iflag=direct 2>/tmp/dd_err.$$ || true
        END=$$(date +%s%N)
        ELAPSED_NS=$$((END - START))
        if [[ $$ELAPSED_NS -gt 0 ]]; then
            ELAPSED_S=$$(awk "BEGIN {{printf \"%.3f\", $$ELAPSED_NS / 1000000000}}")
            READ_MBS=$$(awk "BEGIN {{printf \"%.1f\", ($$SIZE_MB) / $$ELAPSED_S}}")
        else
            READ_MBS="0"
        fi
    else
        READ_MBS="N/A"
    fi

    rm -f "$$TESTPATH" 2>/dev/null || true

    # Get filesystem type for this mount
    FSTYPE=$$(df -T "$$MOUNT" 2>/dev/null | tail -1 | awk '{{print $$2}}' || echo "unknown")

    echo "RESULT mount=$$MOUNT fstype=$$FSTYPE write_mbs=$$WRITE_MBS read_mbs=$$READ_MBS"
done

echo "=== DONE ==="
'''


def _build_bench_script(size_mb: int, count: int) -> str:
    """Build the benchmark script with parameters substituted."""
    s = _BENCH_TEMPLATE.replace("__SIZE_MB__", str(size_mb))
    s = s.replace("__COUNT__", str(count))
    # Unescape doubled braces back to singles for the bash script
    s = s.replace("{{", "{")
    s = s.replace("}}", "}")
    # Unescape doubled dollars back to singles
    s = s.replace("$$", "$")
    return s


@dataclass
class MountResult:
    mount: str
    fstype: str
    write_mbs: float
    read_mbs: float


@dataclass
class ServerResult:
    server_name: str
    error: str | None = None
    mount_results: list[MountResult] = field(default_factory=list)
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
        sr.error = f"Exit {result.exit_status}: {(stderr)[:200]}"
        return sr

    # Parse output
    sr.raw_mounts = ""
    in_mounts = False
    for line in output.splitlines():
        line = line.strip()

        if line.startswith("--- Mounts ---"):
            in_mounts = True
            continue
        if line.startswith("--- Bandwidth"):
            in_mounts = False
            continue
        if in_mounts and line:
            sr.raw_mounts += line + "\n"

        if line.startswith("RESULT "):
            parts = dict(
                item.split("=", 1) for item in line.removeprefix("RESULT ").split()
            )
            try:
                sr.mount_results.append(MountResult(
                    mount=parts["mount"],
                    fstype=parts.get("fstype", "unknown"),
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
    """Print a formatted bandwidth report to stdout."""
    print()
    print("=" * 90)
    print(f"  STORAGE BANDWIDTH REPORT  —  {size_mb}M test file")
    print("=" * 90)

    # Collect all unique mount labels
    all_mounts: dict[str, str] = {}  # mount_path -> display label
    for sr in results:
        for mr in sr.mount_results:
            label = f"{mr.mount} ({mr.fstype})"
            all_mounts[mr.mount] = label
    mount_order = sorted(all_mounts.keys())

    # Header
    header = f"{'Server':<14}"
    for mp in mount_order:
        header += f" | {all_mounts[mp]:^21}"
    print(header)
    print("-" * 90)

    # Per-server rows
    for sr in sorted(results, key=lambda r: r.server_name):
        if sr.error and not sr.mount_results:
            print(f"{'':>3}{sr.server_name:<11} \033[31mERROR: {sr.error}\033[0m")
            continue

        line = f"  {sr.server_name:<12}"
        for mp in mount_order:
            found = next((m for m in sr.mount_results if m.mount == mp), None)
            if found:
                cell = f"W:{_fmt(found.write_mbs)} R:{_fmt(found.read_mbs)}"
            else:
                cell = f"{'':>15}"
            line += f" |{cell}"
        print(line)

    print("-" * 90)

    # Summary: best read/write per mount across all servers
    print("\n--- Per-Mount Summary (best server) ---")
    for mp in mount_order:
        best_write = 0.0
        best_read = 0.0
        best_w_srv = ""
        best_r_srv = ""
        for sr in results:
            for mr in sr.mount_results:
                if mr.mount == mp:
                    if mr.write_mbs > best_write:
                        best_write = mr.write_mbs
                        best_w_srv = sr.server_name
                    if mr.read_mbs > best_read:
                        best_read = mr.read_mbs
                        best_r_srv = sr.server_name
        print(f"  {all_mounts[mp]:<20}  "
              f"Best Write: {_fmt(best_write)} MB/s ({best_w_srv})  "
              f"Best Read:  {_fmt(best_read)} MB/s ({best_r_srv})")

    # Server ranking by avg write speed
    print("\n--- Server Ranking (avg bandwidth across all mounts) ---")
    rankings: list[tuple[str, float, float]] = []
    for sr in results:
        if not sr.mount_results:
            continue
        avg_w = sum(m.write_mbs for m in sr.mount_results) / len(sr.mount_results)
        avg_r = sum(m.read_mbs for m in sr.mount_results) / len(sr.mount_results)
        rankings.append((sr.server_name, avg_w, avg_r))
    rankings.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, avg_w, avg_r) in enumerate(rankings, 1):
        print(f"  {rank:2}. {name:<12}  Avg Write: {_fmt(avg_w)} MB/s  "
              f"Avg Read: {_fmt(avg_r)} MB/s")

    # NAS comparison (if available)
    nas_results = []
    for sr in results:
        for mr in sr.mount_results:
            if "nas" in mr.mount.lower():
                nas_results.append((sr.server_name, mr))
    if nas_results:
        print("\n--- NAS (/mnt/nas) Comparison ---")
        nas_results.sort(key=lambda x: x[1].write_mbs, reverse=True)
        for name, mr in nas_results:
            print(f"  {name:<12}  Write: {_fmt(mr.write_mbs)} MB/s  "
                  f"Read: {_fmt(mr.read_mbs)} MB/s  ({mr.fstype})")

    print("=" * 90)


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
        "--timeout", type=int, default=60,
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
