"""Persistent JSON registry for launched jobs."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import JobRecord, LaunchRecord, Registry

_REGISTRY_DIR = os.path.expanduser("~/.cache/gpu-jobs")
_REGISTRY_PATH = os.path.join(_REGISTRY_DIR, "registry.json")


def _ensure_dir() -> None:
    os.makedirs(_REGISTRY_DIR, exist_ok=True)


def load_registry() -> Registry:
    """Load the registry from disk, or return empty if not found."""
    _ensure_dir()
    if not os.path.exists(_REGISTRY_PATH):
        return Registry()
    try:
        with open(_REGISTRY_PATH, "r") as f:
            data = json.load(f)
        return Registry(**data)
    except Exception:
        return Registry()


def save_registry(reg: Registry) -> None:
    """Persist the registry to disk."""
    _ensure_dir()
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(reg.model_dump(), f, indent=2, default=str)


def add_job(job: JobRecord) -> None:
    """Add or update a job record in the registry."""
    reg = load_registry()
    job.launched_at = job.launched_at or datetime.now(timezone.utc).isoformat()
    reg.jobs[job.name] = job
    save_registry(reg)


def get_job(name: str) -> Optional[JobRecord]:
    """Get a single job record by name."""
    reg = load_registry()
    return reg.jobs.get(name)


def list_jobs(project: Optional[str] = None) -> list[JobRecord]:
    """List all jobs, optionally filtered by project."""
    reg = load_registry()
    jobs = list(reg.jobs.values())
    if project:
        jobs = [j for j in jobs if j.project == project]
    return jobs


def update_task_status(
    job_name: str,
    task_name: str,
    status: str,
    pid: Optional[int] = None,
) -> None:
    """Update the status of a single task within a job."""
    reg = load_registry()
    job = reg.jobs.get(job_name)
    if not job:
        return
    for t in job.tasks:
        if t.name == task_name:
            t.status = status
            if pid is not None:
                t.pid = pid
            break
    save_registry(reg)


def remove_job(name: str) -> bool:
    """Remove a job from the registry. Returns True if it existed."""
    reg = load_registry()
    if name in reg.jobs:
        del reg.jobs[name]
        save_registry(reg)
        return True
    return False
