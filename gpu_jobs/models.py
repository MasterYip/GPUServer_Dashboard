"""Pydantic data models for job configuration and launch records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel


# ── GPU candidate (from scheduler) ──────────────────────────────────────


@dataclass
class GpuCandidate:
    """A free GPU found by the scheduler, ranked by score (higher = better)."""

    server_name: str
    host: str
    port: int
    gpu_index: int
    mem_free_mb: float
    mem_total_mb: float
    gpu_util_pct: float
    score: float = 0.0


# ── Job config models ───────────────────────────────────────────────────


class TaskSpec(BaseModel):
    """A single task within a job — one GPU's worth of work."""

    name: str  # used for screen session name + log filename
    overrides: dict[str, Any] = {}
    gpu_min_memory_mb: int = 4000


class PathRemapRule(BaseModel):
    """Server-name-based path substitution rule.

    If *server_name* starts with ``match`` and is NOT in ``exclude``,
    each key→value in ``map`` is applied as a string replace on the command.
    """

    match: str
    exclude: list[str] = []
    map: dict[str, str] = {}


class JobDefaults(BaseModel):
    """Shared settings merged into every task."""

    work_dir: str = ""
    python: str = "python3"
    cfg: str = ""
    log_dir: str = "/tmp/gpu-jobs"
    env: dict[str, str] = {}
    overrides: dict[str, Any] = {}
    path_remap: list[PathRemapRule] = []
    exp_group: str = ""  # experiment registry group (e.g. group_09_cps_mode_win)


class JobConfig(BaseModel):
    """Top-level job configuration loaded from YAML."""

    name: str
    project: str = "default"
    description: str = ""
    defaults: JobDefaults = JobDefaults()
    command: str = ""  # shell template with {var} placeholders
    tasks: list[TaskSpec]


# ── Registry / launch records ───────────────────────────────────────────


class LaunchRecord(BaseModel):
    """Record of a single launched task."""

    name: str
    server: str
    gpu: int
    screen_session: str
    log_file: str
    pid: Optional[int] = None
    status: str = "running"  # running | done | dead | killed
    launched_at: str = ""


class JobRecord(BaseModel):
    """Record of a launched job (collection of tasks)."""

    name: str
    project: str
    config_path: str
    launched_at: str = ""
    tasks: list[LaunchRecord] = []


class Registry(BaseModel):
    """Persistent job registry."""

    jobs: dict[str, JobRecord] = {}
