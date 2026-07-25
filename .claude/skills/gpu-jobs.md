# GPU Job Orchestrator — Agent Skill

This skill lets agents probe GPU servers, launch training jobs, and manage
running tasks across a multi-server GPU cluster.

## Invocation

```
/gpu-jobs probe [--prefer S1,S2]
/gpu-jobs run <job.yaml> [--dry-run]
/gpu-jobs list [--project X]
/gpu-jobs kill <job> [--task T]
/gpu-jobs tail <job> [--task T] [--lines N]
```

## Environment

```bash
# Create once:
python3 -m venv /tmp/gpu_jobs_venv
/tmp/gpu_jobs_venv/bin/pip install -r requirements.txt

# All commands use this:
PY=/tmp/gpu_jobs_venv/bin/python
cd /home/user/CodeSpace/Utils/UsefulLinuxBashScripts/util_scripts/gpu_server_probe
```

## Standard Workflow

### 1. Probe — Assess Cluster State

```bash
$PY cli.py probe --prefer 4090-3
```

Outputs a ranked table of free GPUs sorted by score (free memory × idle%), with preferred servers boosted 2×. Use this to decide how many GPUs are available before writing a job YAML.

### 2. Write Job YAML

Create a config in `job_configs/<project>/<name>.yaml`:

```yaml
name: my-job
project: myproject
description: "What this job does"

defaults:
  work_dir: /data/masteryip/PegasusMoDye/PDPlanner
  python: /data/conda/envs/pegasusmodye/bin/python
  cfg: g1prdp_cond_diffuse.yaml
  log_dir: /tmp/gpu-jobs
  env:
    OMNI_KIT_ACCEPT_EULA: "YES"
    PYTHONPATH: /data/masteryip/PegasusMoDye/PDPlanner
  overrides:
    training.num_epochs: 50
    training.tqdm_interval_sec: 30
    logging.mode: offline

command: |
  cd {work_dir}
  {python} train.py --cfg {cfg} --exp_name {exp_name} {overrides} >> {log_file} 2>&1

tasks:
  - name: variant-a
    overrides:
      policy.actor.backbone.cross_attn_window: 4
    gpu_min_memory_mb: 6000

  - name: variant-b
    overrides:
      policy.actor.backbone.cross_attn_window: 8
```

**Command template variables**: `{work_dir}`, `{python}`, `{cfg}`, `{log_file}`, `{exp_name}` (= task name), `{overrides}`, `{project}`, `{job_name}`, `{task_name}`.

**Path remapping**: 4090-series compute servers auto-remap `/data/...` → `/mnt/4090-3/...`. 4090-3 (primary) and H20/H200 servers use paths as-written.

### 3. Dry-Run — Verify Before Launch (Mandatory)

```bash
$PY cli.py run job_configs/<project>/<name>.yaml --dry-run
```

Prints the exact command, GPU assignment, and log path for each task. **Always show this output to the user for confirmation before launching.**

### 4. Launch

```bash
$PY cli.py run job_configs/<project>/<name>.yaml
```

Each task launches in a `screen -dmS gpu-<task_name>` session on the assigned server. Log files go to `/tmp/gpu-jobs/<project>/<job_name>/<task>.log`. A registry record is saved to `~/.cache/gpu-jobs/registry.json`.

### 5. Monitor & Manage

```bash
$PY cli.py list --project myproject      # show all jobs
$PY cli.py tail my-job --task variant-a  # tail specific task log
$PY cli.py tail my-job --lines 100       # tail all tasks
$PY cli.py kill my-job                   # kill everything
$PY cli.py kill my-job --task variant-a  # kill one task
```

## Agent Guidelines

- **Always dry-run first** and present the output to the user.
- **Never edit servers_rp.yaml** — it contains production server IPs.
- **Job YAMLs go in `job_configs/<project>/`**, one YAML per job. Keep them checked into git so runs are reproducible.
- **Log paths are deterministic**: `/tmp/gpu-jobs/<project>/<job_name>/<task>.log` on the remote server. Use `cli.py tail` to read them; don't construct SSH commands manually.
- **Kill before re-launching** the same job if you need to restart.
- **Prefer 4090-3** for PegasusMoDye runs (it has the canonical filesystem). Use other 4090 servers (4090-1/2/4/5) only when 4090-3 is full.
- **4090-48-6/7 and H20/H200** servers have their own local storage — they do NOT mount 4090-3's filesystem. Job YAMLs targeting them need different `python` and `work_dir` paths in `defaults`.
- **Only 4090-1 through 4090-5** share the data disk via `/mnt/4090-3/`. Always use `--prefer 4090-3,4090-1,4090-2,4090-4,4090-5` for PegasusMoDye runs.
- **Screen sessions** are named `gpu-<task_name>`. To attach manually: `ssh <server> -t screen -r gpu-<task_name>`.
