"""GPU scheduling — probe servers and rank free GPUs."""

from __future__ import annotations

from dashboard.models import ServerMetrics
from .models import GpuCandidate


def find_free_gpus(
    metrics: dict[str, ServerMetrics],
    min_free_memory_mb: float = 4000,
    max_gpu_util_pct: float = 30.0,
    preferred_servers: list[str] | None = None,
    exclude_gpus: set[tuple[str, int]] | None = None,
) -> list[GpuCandidate]:
    """Scan all server metrics and return ranked free GPU candidates.

    Parameters
    ----------
    metrics : dict[str, ServerMetrics]
        Server name → metrics from probe_all_servers().
    min_free_memory_mb : float
        Minimum free GPU memory in MB to consider a GPU "free".
    max_gpu_util_pct : float
        Maximum GPU utilization % to consider a GPU "free".
    preferred_servers : list[str] | None
        Server names to prioritize (2× score multiplier).
    exclude_gpus : set[tuple[str, int]] | None
        (server_name, gpu_index) pairs to exclude (already assigned).

    Returns
    -------
    list[GpuCandidate]
        Sorted by score descending (best first).
    """
    preferred = set(preferred_servers or [])
    excluded = exclude_gpus or set()
    candidates: list[GpuCandidate] = []

    for name, m in metrics.items():
        if m.error:
            continue
        for gpu in m.gpu_info:
            if (name, gpu.index) in excluded:
                continue
            mem_free = gpu.memory_total_mb - gpu.memory_used_mb
            if mem_free < min_free_memory_mb:
                continue
            if gpu.utilization_gpu > max_gpu_util_pct:
                continue

            score = (mem_free / 1024.0) * (1.0 - gpu.utilization_gpu / 100.0)
            if name in preferred:
                score *= 2.0

            candidates.append(
                GpuCandidate(
                    server_name=name,
                    host="",   # filled in later by resolve_host_port()
                    port=22,
                    gpu_index=gpu.index,
                    mem_free_mb=mem_free,
                    mem_total_mb=gpu.memory_total_mb,
                    gpu_util_pct=gpu.utilization_gpu,
                    score=score,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def resolve_host_port(
    candidates: list[GpuCandidate],
    server_configs: dict[str, tuple[str, int]],
) -> list[GpuCandidate]:
    """Fill in host and port from server configs.

    Parameters
    ----------
    candidates : list[GpuCandidate]
        Candidates with server_name set but host/port may be default.
    server_configs : dict[str, tuple[str, int]]
        Mapping of server name → (host, port).

    Returns
    -------
    list[GpuCandidate]
    """
    for c in candidates:
        if c.server_name in server_configs:
            c.host, c.port = server_configs[c.server_name]
    return candidates
