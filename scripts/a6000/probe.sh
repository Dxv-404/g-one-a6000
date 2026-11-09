#!/usr/bin/env bash
# First contact with a new machine, before anything is installed or run. Read-only: it changes nothing. D-090.
#
#   bash scripts/a6000/probe.sh            prints a report and writes results/probe.txt; send that file back
#
# It answers what could not be known before the machine was available: how many GPUs and which, the driver and
# the CUDA version it supports, whether this is plain Linux, WSL, a container or a cluster with a job scheduler,
# and what CPU, memory, disk, network and tools there are. Every check prints a line whether it succeeds or not, so
# a missing tool reads as "absent" rather than as nothing. If the project environment (.venv) already exists, it
# also asks JAX which devices it sees and runs one tiny compile on each GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results
OUT=results/probe.txt
: >"$OUT"
line() { echo "$*" | tee -a "$OUT"; }
have() { command -v "$1" >/dev/null 2>&1; }

line "probe started $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)"
line "== system"
line "   kernel: $(uname -srm)"
grep -qi microsoft /proc/version 2>/dev/null && line "   WSL: yes" || line "   WSL: no"
[ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null && line "   container: likely" || line "   container: no sign"
[ -r /etc/os-release ] && line "   os: $(. /etc/os-release; echo "$PRETTY_NAME")"
for s in sbatch squeue qsub bsub; do have "$s" && line "   job scheduler command present: $s"; done
[ -n "${SLURM_JOB_ID:-}" ] && line "   inside a Slurm job: $SLURM_JOB_ID"
sudo -n true 2>/dev/null && line "   sudo without password: yes" || line "   sudo without password: no"

line "== gpus"
if have nvidia-smi; then
  nvidia-smi -L 2>&1 | tee -a "$OUT"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.limit,driver_version \
    --format=csv 2>&1 | tee -a "$OUT"
  line "   CUDA version the driver supports: $(nvidia-smi -q 2>/dev/null | grep -m1 -i 'CUDA Version' | cut -d: -f2 | xargs)"
  line "   processes already on the GPUs:"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>&1 | tee -a "$OUT"
else
  line "   nvidia-smi: absent"
fi

line "== cpu, memory, disk"
line "   cpus: $(nproc)  model: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
free -g | tee -a "$OUT"
df -h "$HOME" . 2>&1 | tee -a "$OUT"
line "   load: $(cut -d' ' -f1-3 /proc/loadavg)   users logged in: $(who | wc -l)"

line "== tools and network"
for t in git gh uv python3 tmux curl; do
  flag=--version
  [ "$t" = tmux ] && flag=-V
  if have "$t"; then line "   $t: $("$t" "$flag" 2>&1 | head -1)"; else line "   $t: absent"; fi
done
if curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://github.com >/dev/null 2>&1; then
  line "   github.com reachable: yes"
else
  line "   github.com reachable: no"
fi

if [ -x .venv/bin/python ]; then
  line "== JAX in the project environment"
  .venv/bin/python - <<'EOF' 2>&1 | tee -a "$OUT"
import src  # noqa: F401  the project's runtime settings first
import jax
import jax.numpy as jnp

print("   jax", jax.__version__, "| devices:", [f"{d.platform}:{d.id} {d.device_kind}" for d in jax.devices()])
for d in jax.devices():
    x = jax.device_put(jnp.arange(8.0), d)
    y = jax.jit(lambda v: (v * 3 + 1).sum())(x)
    print(f"   device {d.id}: compiled and ran, result {float(y)}")
EOF
else
  line "== JAX: no project environment yet (.venv absent); the machine check installs it"
fi

line "probe finished $(date '+%H:%M:%S'). Send the file results/probe.txt."
