# Shared Conda Environment Setup Across GPU Servers

All servers mount `/mnt/nas` (NFS 9.8TB) at the same path.
We install Miniconda there so every server sees the same
conda + environments directly.

## Step 1: Install Miniconda on the NAS

Pick **ONE** server (any) and run:

```bash
mkdir -p /mnt/nas/apps
cd /mnt/nas/apps
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /mnt/nas/apps/miniconda3
```

## Step 2: Create a shared conda init script

On the **SAME** server, create `/mnt/nas/apps/conda_init.sh`:

```bash
cat > /mnt/nas/apps/conda_init.sh << 'EOF'
# Source this in ~/.bashrc to activate shared conda
__conda_setup="$('/mnt/nas/apps/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/nas/apps/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/nas/apps/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/nas/apps/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Packages live on NAS too
export CONDA_PKGS_DIRS=/mnt/nas/apps/conda_pkgs
mkdir -p "$CONDA_PKGS_DIRS"
EOF

chmod +r /mnt/nas/apps/conda_init.sh
```

## Step 3: Add to every server's `~/.bashrc`

On **EACH** server, append:

```bash
echo 'source /mnt/nas/apps/conda_init.sh' >> ~/.bashrc
```

Then reload:

```bash
source ~/.bashrc
conda --version   # verify it works
```

## Step 4: Create shared environments

All envs go under `/mnt/nas/apps/miniconda3/envs/` so every server
sees them. From any server:

```bash
conda create -n my_project python=3.10 -y
conda activate my_project
pip install torch numpy ...
```

To create an env at a custom path (useful for per-project envs):

```bash
conda create -p /mnt/nas/envs/my_project python=3.10 -y
conda activate /mnt/nas/envs/my_project
```

## Step 5: Speed tip — keep pkgs dir on NAS

Set this on every server so downloaded packages are cached on
the NAS and reused across servers:

```bash
conda config --system --add pkgs_dirs /mnt/nas/apps/conda_pkgs
```

## Useful conda config for NAS-based setups

```bash
# Prefer the fastest solver
conda config --set solver libmamba

# Don't auto-activate base env (keep things clean)
conda config --set auto_activate_base false
```

## FAQ

**Q: What if NFS is slow?**

The conda binary and env Python files are small. The main cost
is loading large shared libraries (.so). If this is slow,
symlink `/mnt/nas/apps/miniconda3/envs/<name>/lib` to a local
SSD. But in practice, NFS metadata caching makes it fine.

**Q: What about pip-installed packages?**

They go into the env on NAS automatically. Same advice: if
heavy, consider a local symlink for the env's `site-packages`.

**Q: Can I still have per-server conda envs?**

Yes. Just create them in `~/miniconda3/envs/` as usual. Only
envs under `/mnt/nas` are shared.

## One-liner to bootstrap a fresh server

```bash
ssh user@server 'bash -c "echo source /mnt/nas/apps/conda_init.sh >> ~/.bashrc && source /mnt/nas/apps/conda_init.sh && conda --version"'
```
