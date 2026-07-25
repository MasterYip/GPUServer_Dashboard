# Plan: GPU Job Orchestrator (gpu-jobs)

## Context

Launching training runs across the 7-server/56-GPU cluster is currently manual and error-prone: SSH per server, `nvidia-smi`, mentally assign GPUs, construct `screen` commands with correct cross-server paths. We've had multiple failures from GPU conflicts, wrong paths, corrupted configs, and CPU fallback.

The user wants:
1. **Auto GPU discovery** — probe all servers, rank free GPUs
2. **Job launch** — assign tasks to best GPUs, launch via screen
3. **Job management** — list, kill, tail logs of running jobs
4. **Cross-project** — not tied to PDPlanner; any project can define job YAMLs
5. **Agent-friendly logs** — structured log paths so agents can read output without parsing project-specific formats
6. **SKILL interface** — `/gpu-jobs` so Claude Code agents can operate it

## Existing Infrastructure to Reuse

- **`gpu_server_probe/`** at `/home/user/CodeSpace/Utils/UsefulLinuxBashScripts/util_scripts/gpu_server_probe/`
  - `dashboard/probe.py` — `probe_all_servers()` via `asyncssh`
  - `dashboard/config.py` — `load_servers()` from YAML
  - `dashboard/models.py` — Pydantic `ServerConfig`, `GpuInfo`, `ServerMetrics`
  - `servers_rp.yaml` — complete server list (H20, H200, 4090 series)
  - `run_dashboard.py` — argparse subcommand CLI pattern

- **PegasusMoDye SOP** at `doc/pegasusmodye_agent_sop.md` — canonical training command format for this project

## New Files

All in `/home/user/CodeSpace/Utils/UsefulLinuxBashScripts/util_scripts/gpu_server_probe/`:

```
gpu_server_probe/
├── gpu_jobs/
│   ├── __init__.py
│   ├── scheduler.py          # GPU scoring & ranking
│   ├── job_config.py          # Generic job YAML loader (project-agnostic)
│   ├── launcher.py            # SSH screen launcher + kill + list + tail
│   ├── registry.py            # Local SQLite/JSON registry of launched jobs
│   └── models.py              # Pydantic models: JobConfig, TaskSpec, LaunchRecord
├── cli.py                     # Unified CLI: probe | run | dry-run | list | kill | tail
├── job_configs/               # Project job YAML configs live here
│   └── pegasusmodye/          # Per-project subdirectory
│       └── example_ablation.yaml
└── (existing dashboard/ files untouched)
```

SKILL file:
```
/home/user/CodeSpace/Diffusion/PegasusMoDye/.claude/skills/gpu-jobs.md
```

## Detailed Design

### 1. Job Config YAML — Project-Agnostic Format

Instead of baking in PDPlanner assumptions, the YAML defines a **generic command template** per task. The program doesn't need to understand the command — it just substitutes variables and launches.

```yaml
# job_configs/pegasusmodye/ablation_cps_mode_win.yaml
name: cps-mode-win-ablation
project: pegasusmodye

# These vars are available in command templates as {var_name}
defaults:
  work_dir: /mnt/4090-3/masteryip/PegasusMoDye/PDPlanner
  python: /mnt/4090-3/conda/envs/pegasusmodye/bin/python
  cfg: g1prdp_cond_diffuse.yaml
  log_dir: /tmp/gpu-jobs                    # centralized log root
  env:                                       # env vars set before command
    OMNI_KIT_ACCEPT_EULA: "YES"
    PYTHONPATH: /mnt/4090-3/masteryip/PegasusMoDye/PDPlanner

# Command template — {overrides} is auto-generated from per-task overrides
# {log_file} is auto-set to {log_dir}/{project}/{job_name}/{task_name}.log
command: |
  cd {work_dir}
  {python} train.py --cfg {cfg} --exp_name {exp_name} {overrides} >> {log_file} 2>&1

tasks:
  - name: cps-T_state_win4         # used for screen session name + log filename
    overrides:
      policy.actor.clean_past_state: true
      policy.actor.backbone.cross_attn_mode: state
      policy.actor.backbone.cross_attn_window: 4
    gpu_min_memory_mb: 6000        # optional: task-specific GPU requirement

  - name: cps-T_state_global
    overrides:
      policy.actor.clean_past_state: true
      policy.actor.backbone.cross_attn_mode: state
      policy.actor.backbone.cross_attn_window: null
```

**Key design decisions**:
- `command` is a shell template — the program just does `{var}` substitution, nothing project-specific
- `{overrides}` is auto-built from `defaults.overrides` merged with per-task `overrides`, formatted as `key=value key=value`
- `{log_file}` auto-generated: `{log_dir}/{project}/{job_name}/{task_name}.log`
- For projects with a different script (e.g. `train_exp_mode.py`), the user writes a different command template
- The program NEVER parses training output — it only manages logs at the filesystem level

### 2. GPU Scheduler (`gpu_jobs/scheduler.py`)

```python
@dataclass
class GpuCandidate:
    server_name: str
    host: str
    port: int
    gpu_index: int
    mem_free_mb: float
    mem_total_mb: float
    gpu_util_pct: float
    score: float

def find_free_gpus(
    metrics: dict[str, ServerMetrics],
    min_free_memory_mb: float = 4000,
    max_gpu_util_pct: float = 30.0,
    preferred_servers: list[str] | None = None,
    exclude_gpus: set[tuple[str, int]] | None = None,  # already-assigned GPUs
) -> list[GpuCandidate]:
```

Scoring: `score = mem_free_gb * (1.0 - gpu_util_pct / 100)`, with 2× multiplier for preferred servers.

### 3. Job Registry (`gpu_jobs/registry.py`)

A simple JSON file at `~/.cache/gpu-jobs/registry.json` tracking all launched jobs:

```json
{
  "jobs": {
    "cps-mode-win-ablation": {
      "project": "pegasusmodye",
      "config_path": "job_configs/pegasusmodye/ablation_cps_mode_win.yaml",
      "launched_at": "2026-07-25T12:00:00",
      "tasks": [
        {
          "name": "cps-T_state_win4",
          "server": "4090-3",
          "gpu": 0,
          "screen_session": "gpu-cps-T_state_win4",
          "log_file": "/tmp/gpu-jobs/pegasusmodye/cps-mode-win-ablation/cps-T_state_win4.log",
          "pid": 1452392,
          "status": "running"
        }
      ]
    }
  }
}
```

This enables `list`, `kill`, and `tail` without re-probing servers. Status is refreshed on demand by checking if screen sessions / PIDs are still alive.

### 4. CLI (`cli.py`)

```
gpu-jobs probe [--prefer 4090-3,4090-5] [--min-mem 6000]
    Probe all servers, print ranked free GPU table

gpu-jobs run <config.yaml> [--dry-run] [--max-gpus N] [--prefer S1,S2]
    Probe → assign → launch. --dry-run prints assignments without launching.

gpu-jobs list [--project X] [--status running|done|dead]
    Show all tracked jobs and their task status from registry

gpu-jobs kill <job_name> [--task <task_name>]
    Kill screen sessions for a job (or specific task) via SSH

gpu-jobs tail <job_name> [--task <task_name>] [--lines 20]
    SSH tail the log file(s) for a job — agent-friendly log reading

gpu-jobs clean [--job <name>] [--older-than 7d]
    Remove dead jobs from registry, optionally kill orphan screen sessions
```

### 5. Launcher (`gpu_jobs/launcher.py`)

Four async operations, all using `asyncssh`:

**`launch_tasks(tasks, gpus, config) -> list[LaunchRecord]`**
1. For each task, pick the best GPU from `find_free_gpus()`
2. Mark that GPU as taken (exclude from subsequent assignments)
3. Build full command via template substitution
4. SSH → `mkdir -p {log_dir}/...` → `screen -dmS gpu-{task_name} bash -c '{command}'`
5. Record to registry

**`kill_task(screen_name, server) -> bool`**
1. SSH → `screen -S {screen_name} -X quit`
2. Update registry status

**`list_screens(server) -> list[str]`**
1. SSH → `screen -ls` (parse output)
2. Cross-reference with registry

**`tail_log(log_path, server, lines=20) -> str`**
1. SSH → `tail -n {lines} {log_path}`
2. Return stdout directly (agent-readable)

### 6. Log Structure

Centralized, agent-friendly:

```
/tmp/gpu-jobs/
├── pegasusmodye/
│   ├── cps-mode-win-ablation/
│   │   ├── cps-T_state_win4.log
│   │   ├── cps-T_state_global.log
│   │   └── ...
│   └── cross-attn-window/
│       ├── win0.log
│       └── ...
├── other_project/
│   └── ...
└── _registry.json      # symlink to ~/.cache/gpu-jobs/registry.json
```

Benefits:
- Agents can `Read /tmp/gpu-jobs/pegasusmodye/<job>/<task>.log` without SSH
- Log paths are deterministic from job config, not random
- One log per task, not interleaved — easy to grep

**Note**: On non-primary servers, `/tmp/gpu-jobs/` exists on the remote server's local `/tmp`, so agents must SSH to read logs. But the path is still deterministic: `ssh <server> tail -100 /tmp/gpu-jobs/<project>/<job>/<task>.log`.

### 7. SKILL (`gpu-jobs.md`)

Placed at `/home/user/CodeSpace/Diffusion/PegasusMoDye/.claude/skills/gpu-jobs.md`.

Workflow for agents:
1. User says "launch 8 ablation runs"
2. Agent writes a job YAML to `gpu_server_probe/job_configs/pegasusmodye/<name>.yaml`
3. Agent runs `cli.py dry-run <config>` — shows GPU assignments, asks user to confirm
4. Agent runs `cli.py run <config>`
5. Agent can later `cli.py list`, `cli.py tail`, `cli.py kill`

---

## Data Flow

```
User: "Launch ablation X with N tasks"
    ↓
Agent: writes job YAML (or reuses existing template)
Agent: `cli.py dry-run job.yaml` → sees GPU assignment
    ↓ (user confirms)
Agent: `cli.py run job.yaml`
    ↓
    1. probe_all_servers() → ServerMetrics
    2. find_free_gpus() → ranked candidates
    3. For each task: assign best GPU, build command, SSH screen launch
    4. Write registry
    5. Print summary table:
       ┌──────────────────────┬──────────┬──────┬─────────────────────────────────┐
       │ Task                 │ Server   │ GPU  │ Log                             │
       ├──────────────────────┼──────────┼──────┼─────────────────────────────────┤
       │ cps-T_state_win4     │ 4090-3   │ 0    │ /tmp/gpu-jobs/pegasusmodye/... │
       │ cps-T_state_global   │ 4090-3   │ 5    │ /tmp/gpu-jobs/pegasusmodye/... │
       └──────────────────────┴──────────┴──────┴─────────────────────────────────┘

Later:
Agent: `cli.py list --project pegasusmodye` → shows status
Agent: `cli.py tail cps-mode-win-ablation --task cps-T_state_win4` → reads log
Agent: `cli.py kill cps-mode-win-ablation` → kills all screen sessions
```

---

## Verification

1. `cli.py probe` — prints ranked GPU table from live servers
2. `cli.py dry-run example_ablation.yaml` — shows command strings and GPU assignment without SSH launch
3. `cli.py run example_ablation.yaml` with 1 task on free GPU → verify screen session created, log file appears
4. `cli.py list` → shows running job
5. `cli.py tail <job> --task <task>` → shows training output
6. `cli.py kill <job>` → screen session terminated, registry updated
7. Agent invokes `/gpu-jobs` → writes YAML, dry-runs, launches, tails — end to end
