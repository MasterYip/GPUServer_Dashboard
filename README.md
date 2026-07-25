# GPU Server Probe — Resource Dashboard + Job Orchestrator

Real-time GPU, CPU, and RAM monitoring dashboard for multiple servers via SSH.
Two modes: **web dashboard** (browser) and **terminal TUI** (Rich).

Also includes a **GPU Job Orchestrator** (`cli.py`) for probing free GPUs, launching
training jobs, and managing running tasks across the cluster — see [Part 2](#part-2-gpu-job-orchestrator-clipy) below.

<div align="center">
  <video src="https://github.com/user-attachments/assets/9b05aaf0-dcb5-4ed3-95b7-d51de3671ef8"
         controls muted autoplay loop
         width="100%"
         style="max-width: 100%; border-radius: 8px;">
  </video>
</div>

---

## Part 1: Resource Dashboard

### Prerequisites

- Python 3.10+
- SSH key-based access to all target servers (the servers must have your public key in `~/.ssh/authorized_keys`)

### Installation

```bash
cd util_scripts/gpu_server_probe
pip install -r requirements.txt
```

### Configuration

Edit `servers.yaml` to match your setup. The `defaults` block provides fallback values:

```yaml
defaults:
  user: your_ssh_user           # default SSH user
  identity_file: ~/.ssh/id_rsa  # default SSH key

servers:
  - name: MyServer-1
    host: 192.168.1.10
    port: 22                    # optional, defaults to 22
    # user and identity_file inherited from defaults

  - name: MyServer-2
    host: 192.168.1.11
    port: 22222
    user: custom_user           # override default user
    identity_file: ~/.ssh/custom_key  # override default key
```

The existing `servers.yaml` is pre-populated from `server.md`.

### Usage

#### Web Dashboard

```bash
# Start on http://localhost:8080
./run_dashboard.py web --config servers_rp.yaml

# Custom port and probe interval
./run_dashboard.py web --port 9090 --interval 5 --ssh-timeout 8
```

Open your browser to the displayed address. The dashboard auto-refreshes via
Server-Sent Events — no manual refresh needed.

#### Auto-start on Boot (systemd)

```bash
# Install as a user service that starts automatically at boot
./scripts/install_service.sh

# Customize: port, config, interval
./scripts/install_service.sh --config servers_rp.yaml --port 9090 --interval 3

# Restart after config changes
./scripts/restart_service.sh

# Check status / view logs
systemctl --user status gpu-dashboard
journalctl --user -u gpu-dashboard -f

# Remove the service
./scripts/uninstall_service.sh
```

The service uses `loginctl enable-linger` so it starts at boot even before login.

#### Terminal TUI

```bash
./run_dashboard.py tui --config servers_rp.yaml

# Faster refresh
./run_dashboard.py tui --interval 5
```

Press `Ctrl+C` to exit.

#### Full CLI Help

```bash
./run_dashboard.py --help
./run_dashboard.py web --help
./run_dashboard.py tui --help
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `servers.yaml` (next to script) | Path to server config |
| `--interval N` | `10` | Seconds between probe cycles |
| `--ssh-timeout N` | `5` | Per-server SSH timeout (seconds) |
| `--verbose, -v` | off | Enable debug logging |
| `--port N` (web only) | `8080` | HTTP listen port |
| `--host H` (web only) | `0.0.0.0` | HTTP bind address |

### How It Works

1. Reads server list from `servers.yaml`
2. Every `--interval` seconds, probes all servers **in parallel** via `asyncssh`
3. On each server, runs:
   - `nvidia-smi` for GPU utilization, memory, temperature, power
   - `top -bn2` for CPU usage
   - `free -b` for RAM usage
4. Results are displayed in a dark-themed dashboard with color-coded bars

### Dashboard Features

- 🟢🟡🔴 **Color-coded status**: green (<60%), yellow (<85%), red (≥85%)
- **GPU**: per-GPU utilization bar, memory used/total, temperature, power draw
- **CPU**: aggregate percentage with bar
- **RAM**: used/total GB with bar
- **Error handling**: unreachable servers show error cards with last-known data

---

## Part 2: GPU Job Orchestrator (`cli.py`)

Probe, schedule, launch, and manage GPU training jobs across all servers in one command.
Replaces manual `ssh` + `nvidia-smi` + `screen` per server.

### Quick Start

```bash
# Create a venv once (or reuse the project's existing environment):
python3 -m venv /tmp/gpu_jobs_venv
/tmp/gpu_jobs_venv/bin/pip install -r requirements.txt

cd util_scripts/gpu_server_probe
PY=/tmp/gpu_jobs_venv/bin/python
```

### Commands

```bash
# Probe — see all free GPUs ranked by score
$PY cli.py probe
$PY cli.py probe --prefer 4090-3,4090-5

# Run — launch a job from a YAML config
$PY cli.py run job_configs/<project>/<name>.yaml --dry-run    # always preview first
$PY cli.py run job_configs/<project>/<name>.yaml              # live launch
$PY cli.py run job_configs/<project>/<name>.yaml --max-gpus 8 --prefer 4090-3

# List — show tracked jobs
$PY cli.py list --project pegasusmodye

# Kill — stop a job (or single task)
$PY cli.py kill <job_name> --config-servers servers_rp.yaml
$PY cli.py kill <job_name> --task <task_name>

# Tail — read task logs via SSH
$PY cli.py tail <job_name> --lines 50 --config-servers servers_rp.yaml
$PY cli.py tail <job_name> --task <task_name>
```

### Job YAML Format

Job configs are project-agnostic YAML files in `job_configs/<project>/<name>.yaml`.
See `job_configs/pegasusmodye/example_ablation.yaml` for a complete example.

```yaml
name: my-job                           # used in registry + log paths
project: myproject
description: "What this job does"      # optional

defaults:
  work_dir: /data/masteryip/PegasusMoDye/PDPlanner
  python: /data/conda/envs/pegasusmodye/bin/python
  cfg: g1prdp_cond_diffuse.yaml
  log_dir: /tmp/gpu-jobs
  env:                                 # env vars set before command
    OMNI_KIT_ACCEPT_EULA: "YES"
    PYTHONPATH: /data/masteryip/PegasusMoDye/PDPlanner
  overrides:                           # shared CLI overrides, merged per-task
    training.num_epochs: 50
    training.tqdm_interval_sec: 30
    logging.mode: offline

command: |                             # shell template — {var} substitution
  cd {work_dir}
  {python} train.py --cfg {cfg} --exp_name {exp_name} {overrides} >> {log_file} 2>&1

tasks:
  - name: variant-a                    # screen session: gpu-variant-a
    overrides:                         # per-task overrides (merged with defaults)
      policy.actor.backbone.cross_attn_window: 4
    gpu_min_memory_mb: 6000            # optional

  - name: variant-b
    overrides:
      policy.actor.backbone.cross_attn_window: 8
```

#### Template Variables

| Variable | Source |
|----------|--------|
| `{work_dir}`, `{python}`, `{cfg}`, `{log_dir}` | `defaults` block |
| `{log_file}` | Auto-generated: `{log_dir}/{project}/{job_name}/{task_name}.log` |
| `{exp_name}` | `task.name` |
| `{overrides}` | `defaults.overrides` + `task.overrides` merged, formatted `key=value ...` |
| `{project}`, `{job_name}`, `{task_name}` | From config / task name |

#### Path Remapping

4090-series compute servers (4090-1/2/4/5, 4090-48-6/7) auto-remap
`/data/...` → `/mnt/4090-3/...`. 4090-3 (primary) and H20/H200 servers
use paths as-written in the YAML.

### How Launch Works

1. **Probe** — `asyncssh` to all servers in `servers_rp.yaml`, run `nvidia-smi`
2. **Rank** — free GPUs scored by: free memory GB × (1 − utilization%), preferred servers 2×
3. **Assign** — best GPU per task; no two tasks share a GPU in one launch
4. **Launch** — SSH → `mkdir -p {log_dir}` → `screen -dmS gpu-{task_name} bash -c '{command}'`
5. **Record** — job entry saved to `~/.cache/gpu-jobs/registry.json`

### Log Files

```
/tmp/gpu-jobs/
├── pegasusmodye/
│   └── <job_name>/
│       ├── variant-a.log
│       └── variant-b.log
└── _registry.json
```

Log paths are deterministic — agents can predict them from the job YAML without
querying the registry.

### Standard Workflow (Agents)

1. **Probe** → `cli.py probe` to see available GPUs
2. **Write YAML** → `job_configs/<project>/<name>.yaml`
3. **Dry-run** → `cli.py run ... --dry-run` and present output to user
4. **Launch** → `cli.py run ...` after user confirms
5. **Monitor** → `cli.py tail <job>` to read logs, `cli.py list` to check status
6. **Clean up** → `cli.py kill <job>` when done

A SKILL file is provided at `.claude/skills/gpu-jobs.md` for Claude Code agents.

---

## File Structure

```
gpu_server_probe/
├── server.md               # Server list reference (human-readable)
├── servers.yaml             # Machine-readable server config (dev)
├── servers_rp.yaml          # Production server config
├── requirements.txt         # Python dependencies
├── run_dashboard.py         # Dashboard CLI entry point
├── cli.py                   # Job Orchestrator CLI entry point
├── README.md                # This file
├── .claude/
│   └── skills/
│       └── gpu-jobs.md      # Agent SKILL definition
├── gpu_jobs/                # Job orchestrator package
│   ├── __init__.py
│   ├── models.py            # JobConfig, TaskSpec, GpuCandidate, LaunchRecord
│   ├── scheduler.py         # GPU scoring & ranking
│   ├── job_config.py        # YAML loader + command template substitution
│   ├── launcher.py          # asyncssh screen launch / kill / tail
│   └── registry.py          # JSON job registry
├── job_configs/             # Per-project job YAML configs
│   └── pegasusmodye/
│       └── example_ablation.yaml
├── dashboard/               # Resource dashboard package
│   ├── __init__.py
│   ├── models.py            # Pydantic data models
│   ├── config.py            # YAML config loader
│   ├── probe.py             # asyncssh parallel probing
│   ├── web.py               # FastAPI + SSE web backend
│   ├── tui.py               # Rich terminal dashboard
│   └── templates/
│       └── dashboard.html   # Self-contained web UI
├── scripts/                 # systemd service helpers
│   ├── install_service.sh
│   ├── restart_service.sh
│   └── uninstall_service.sh
├── doc/
│   └── shared_conda_setup.md
├── docs/
└── tests/
```
