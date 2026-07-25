"""Load and validate job YAML configs, with template substitution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .models import JobConfig, JobDefaults, TaskSpec


def load_job_config(path: str) -> JobConfig:
    """Load a job YAML config file, with defaults merging.

    Parameters
    ----------
    path : str
        Path to the job YAML file.

    Returns
    -------
    JobConfig
        Fully resolved job config with defaults merged into each task.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty job config: {path}")

    defaults_raw = raw.get("defaults", {})
    defaults = JobDefaults(**defaults_raw)

    tasks = []
    for t in raw.get("tasks", []):
        # Merge defaults.overrides → per-task overrides (task wins)
        merged_overrides = dict(defaults.overrides)
        merged_overrides.update(t.get("overrides", {}))
        t_copy = dict(t)
        t_copy["overrides"] = merged_overrides
        # Fill in gpu_min_memory_mb from task or default 4000
        t_copy.setdefault("gpu_min_memory_mb", 4000)
        tasks.append(TaskSpec(**t_copy))

    config = JobConfig(
        name=raw.get("name", Path(path).stem),
        project=raw.get("project", "default"),
        description=raw.get("description", ""),
        defaults=defaults,
        command=raw.get("command", ""),
        tasks=tasks,
    )
    return config


def build_task_command(
    config: JobConfig,
    task: TaskSpec,
    log_file: str,
    server_name: str = "",
) -> str:
    """Build the shell command string for a single task.

    Parameters
    ----------
    config : JobConfig
    task : TaskSpec
    log_file : str
        Full path for the task's log file.
    server_name : str
        Server name. Path remapping is applied from config.defaults.path_remap
        (first matching rule wins).

    Returns
    -------
    str
        Shell command string ready for screen -dmS bash -c '...'.
    """
    # Build overrides string
    overrides_str = " ".join(
        f"{k}={_fmt_override(v)}" for k, v in task.overrides.items()
    )

    # Template variables
    vars_dict: dict[str, str] = {
        "work_dir": config.defaults.work_dir,
        "python": config.defaults.python,
        "cfg": config.defaults.cfg,
        "log_dir": config.defaults.log_dir,
        "log_file": log_file,
        "exp_name": task.name,
        "overrides": overrides_str,
        "project": config.project,
        "job_name": config.name,
        "task_name": task.name,
    }

    cmd = config.command
    for key, val in vars_dict.items():
        cmd = cmd.replace("{" + key + "}", str(val))

    # Apply path remapping rules from YAML config (first match wins)
    for rule in config.defaults.path_remap:
        if server_name.startswith(rule.match) and server_name not in rule.exclude:
            for find, replace in rule.map.items():
                cmd = cmd.replace(find, replace)
            break

    return cmd.strip()


def _fmt_override(value: Any) -> str:
    """Format an override value for CLI use."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_log_path(
    log_dir: str,
    project: str,
    job_name: str,
    task_name: str,
) -> str:
    """Build the deterministic log file path for a task.

    Returns
    -------
    str
        {log_dir}/{project}/{job_name}/{task_name}.log
    """
    return os.path.join(log_dir, project, job_name, f"{task_name}.log")
