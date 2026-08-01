#!/usr/bin/env python3
"""GPU Job Orchestrator — probe, launch, and manage GPU jobs across servers.

Usage:
  gpu-jobs probe [--prefer S1,S2] [--min-mem MB]
  gpu-jobs run <job.yaml> [--dry-run] [--max-gpus N] [--prefer S1,S2]
  gpu-jobs list [--project X]
  gpu-jobs kill <job_name> [--task T] [--config-servers PATH]
  gpu-jobs tail <job_name> [--task T] [--lines N] [--config-servers PATH]

The --config-servers flag points to the servers YAML file (defaults to
servers_rp.yaml next to this script when needed).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _default_servers() -> str:
    return str(HERE / "servers_rp.yaml")


def _servers_required(args) -> str:
    """Return servers config path, or exit with helpful message."""
    path = getattr(args, "config_servers", None) or _default_servers()
    if not os.path.exists(path):
        print(f"Servers config not found: {path}")
        print("Specify with --config-servers, or place servers_rp.yaml next to cli.py")
        sys.exit(1)
    return path


# ── Subcommand handlers ─────────────────────────────────────────────────


def cmd_probe(args: argparse.Namespace) -> None:
    """Probe all servers and print ranked free GPU table."""
    from gpu_jobs.launcher import probe_and_rank

    preferred = None
    if args.prefer:
        preferred = [s.strip() for s in args.prefer.split(",")]

    allowed = None
    if getattr(args, "only_servers", None):
        allowed = {s.strip() for s in args.only_servers.split(",")}

    async def _run():
        candidates = await probe_and_rank(
            _default_servers(),
            preferred_servers=preferred,
            allowed_servers=allowed,
            ssh_timeout=args.ssh_timeout,
        )
        if not candidates:
            print("No free GPUs found.")
            return

        print(f"\n{'Server':<16} {'GPU':>4} {'Free Mem':>9} {'Total':>9} {'Util':>6} {'Score':>8}")
        print("-" * 60)
        for c in candidates:
            if args.min_mem and c.mem_free_mb < args.min_mem:
                continue
            util_str = f"{c.gpu_util_pct:.0f}%"
            print(
                f"{c.server_name:<16} {c.gpu_index:>4} "
                f"{c.mem_free_mb:>7.0f} MB {c.mem_total_mb:>7.0f} MB "
                f"{util_str:>6} {c.score:>8.1f}"
            )
        print(f"\n{candidates[0].mem_free_mb - candidates[-1].mem_free_mb:.0f} MB "
              f"spread from best to worst.  {len(candidates)} GPUs available.\n")

    asyncio.run(_run())


def cmd_run(args: argparse.Namespace) -> None:
    """Launch a job from YAML config."""
    from gpu_jobs.job_config import load_job_config
    from gpu_jobs.launcher import launch_job

    config_path = args.job_yaml
    if not os.path.exists(config_path):
        print(f"Job config not found: {config_path}")
        sys.exit(1)

    job = load_job_config(config_path)
    print(f"\nJob: {job.name}  ({len(job.tasks)} tasks, project={job.project})")
    if args.dry_run:
        print("Mode: DRY-RUN (no actual launch)\n")
    else:
        print(f"Mode: LIVE\n")

    preferred = None
    if args.prefer:
        preferred = [s.strip() for s in args.prefer.split(",")]

    allowed = None
    if getattr(args, "only_servers", None):
        allowed = {s.strip() for s in args.only_servers.split(",")}

    async def _run():
        record = await launch_job(
            job,
            server_config_path=_default_servers(),
            dry_run=args.dry_run,
            max_gpus=args.max_gpus,
            preferred_servers=preferred,
            allowed_servers=allowed,
            ssh_timeout=args.ssh_timeout,
            no_preflight=args.no_preflight,
        )

        # Print summary
        print(f"\n{'='*70}")
        print(f"Summary: {job.name}")
        print(f"{'Task':<30} {'Server':<14} {'GPU':>4} {'Status':<10}")
        print("-" * 70)
        for t in record.tasks:
            print(f"{t.name:<30} {t.server:<14} {t.gpu:>4} {t.status:<10}")
        print()

    asyncio.run(_run())


def cmd_list(args: argparse.Namespace) -> None:
    """List tracked jobs from the registry."""
    from gpu_jobs.registry import list_jobs

    jobs = list_jobs(project=args.project)
    if not jobs:
        print("No jobs in registry.")
        return

    for j in jobs:
        print(f"\nJob: {j.name}  project={j.project}  launched={j.launched_at[:19]}")
        print(f"{'Task':<30} {'Server':<14} {'GPU':>4} {'Status':<10} {'Log'}")
        print("-" * 80)
        for t in j.tasks:
            log_short = t.log_file[-50:] if len(t.log_file) > 50 else t.log_file
            print(f"{t.name:<30} {t.server:<14} {t.gpu:>4} {t.status:<10} {log_short}")
    print()


def cmd_kill(args: argparse.Namespace) -> None:
    """Kill a job or specific task."""
    from gpu_jobs.launcher import kill_job
    servers = _servers_required(args) if hasattr(args, 'config_servers') else None

    async def _run():
        ok = await kill_job(
            args.job_name,
            task_name=args.task,
            server_config_path=servers,
        )
        if not ok:
            sys.exit(1)

    asyncio.run(_run())


def cmd_tail(args: argparse.Namespace) -> None:
    """Tail logs for a job."""
    from gpu_jobs.launcher import tail_log
    servers = _servers_required(args) if hasattr(args, 'config_servers') else None

    async def _run():
        text = await tail_log(
            args.job_name,
            task_name=args.task,
            lines=args.lines,
            server_config_path=servers,
        )
        print(text)

    asyncio.run(_run())


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU Job Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", help="Subcommand")

    # --- probe ---
    p_probe = subs.add_parser("probe", help="Probe servers, show free GPUs")
    p_probe.add_argument("--prefer", help="Comma-separated preferred server names")
    p_probe.add_argument("--min-mem", type=float, default=0, help="Min free memory (MB)")
    p_probe.add_argument("--only-servers", help="Comma-separated server names to restrict to")
    p_probe.add_argument("--ssh-timeout", type=int, default=8)

    # --- run ---
    p_run = subs.add_parser("run", help="Launch a job from YAML config")
    p_run.add_argument("job_yaml", help="Path to job YAML config")
    p_run.add_argument("--dry-run", action="store_true", help="Print assignments, don't launch")
    p_run.add_argument("--max-gpus", type=int, default=999, help="Max GPUs to use")
    p_run.add_argument("--prefer", help="Comma-separated preferred server names")
    p_run.add_argument("--only-servers", help="Comma-separated server names to restrict to")
    p_run.add_argument("--ssh-timeout", type=int, default=10)
    p_run.add_argument("--no-preflight", action="store_true", help="Skip pre-flight server checks")

    # --- list ---
    p_list = subs.add_parser("list", help="List tracked jobs")
    p_list.add_argument("--project", help="Filter by project name")

    # --- kill ---
    p_kill = subs.add_parser("kill", help="Kill a job or task")
    p_kill.add_argument("job_name", help="Job name to kill")
    p_kill.add_argument("--task", help="Specific task name within the job")
    p_kill.add_argument("--config-servers", help="Path to servers YAML (for SSH)")

    # --- tail ---
    p_tail = subs.add_parser("tail", help="Tail job log files")
    p_tail.add_argument("job_name", help="Job name")
    p_tail.add_argument("--task", help="Specific task name")
    p_tail.add_argument("--lines", type=int, default=20, help="Lines to tail (default: 20)")
    p_tail.add_argument("--config-servers", help="Path to servers YAML (for SSH)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "probe": cmd_probe,
        "run": cmd_run,
        "list": cmd_list,
        "kill": cmd_kill,
        "tail": cmd_tail,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
