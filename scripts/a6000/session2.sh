#!/usr/bin/env bash
# Pre-run checklist items 4.4 and 4.5 plus the machine check, on the A6000, with the FINAL long-run package.
# docs/increment1-prerun-checklist.md.
#
#   bash scripts/a6000/session2.sh            all three, in order. About 2 to 3 hours in total
#   bash scripts/a6000/session2.sh check      the machine check on this code; run.sh refuses to start without it
#   bash scripts/a6000/session2.sh determinism  4.2 again, on the FINAL program: one seed in 6 fresh processes
#   bash scripts/a6000/session2.sh concurrency  4.4: how many long-run processes at once, on the final program
#   bash scripts/a6000/session2.sh recheck    4.5: S4 on the frozen setting, with the widened network (D-085)
#
# Run inside tmux. Nothing in runs/sweep, runs/replay or runs/longrun is touched. Everything goes into
# runs/session2 and runs/s4-recheck, and at the end into the single file results/session2.tar, which is what to
# send back. Refuses to start while a sweep, a long run, a replay or a calibration is running here.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY="$ROOT/.venv/bin/python"
OUT=runs/session2
STAGE=${1:-all}
mkdir -p "$OUT" results
LOG="$OUT/session2.log"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  session2.sh  $*" | tee -a "$LOG"; }

case "$STAGE" in
all | check | determinism | concurrency | recheck) ;;
*) echo "usage: bash scripts/a6000/session2.sh [all|check|determinism|concurrency|recheck]"; exit 2 ;;
esac

. scripts/a6000/env.sh
scrub_environment >"$OUT/environment.txt"
while read -r line; do say "$line"; done <"$OUT/environment.txt"

[ -x "$PY" ] || { say "no environment at .venv: run  bash scripts/a6000/check.sh  once first."; exit 2; }
if ! sha256sum -c --quiet COPY-MANIFEST.sha256 >/dev/null 2>&1; then
  say "STOPPED: a shipped file has been changed. Undo it with  git checkout -- .  and run again."
  exit 2
fi
if pgrep -f "scripts/measure/(sweep|replay_final|calibrate_s4)\.py" >/dev/null 2>&1; then
  say "STOPPED: a sweep, long run, replay or calibration is running on this machine. Wait for it, then run again."
  exit 2
fi
if ! gpu_probe "$PY" >"$OUT/gpu_probe.txt" 2>&1; then
  tail -n 3 "$OUT/gpu_probe.txt" | tee -a "$LOG"
  say "STOPPED: JAX cannot use the GPU from this terminal. Check that nvidia-smi works, then run again."
  exit 2
fi
say "$(tail -n 1 "$OUT/gpu_probe.txt")"
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
  say "WARNING: not inside tmux. If this terminal closes, the checks stop. Continuing in 10 seconds."
  sleep 10
fi
say "frozen world: $("$PY" -c 'from configs import frozen_world as f; print(f.WORLD, "from", f.SOURCE)')"
say "rough durations: machine check 45 to 75 minutes, concurrency 30 to 60, S4 re-check about 25 (64 episodes, D-092)"
say "GPU right now: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null)"

FAILED=0
step() { # step <name> <command...>
  local name=$1
  shift
  say "starting $name"
  "$@" 2>&1 | tee -a "$OUT/$name.out"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ]; then
    say "$name finished"
  else
    say "$name exited with $rc; the others still run. Its output is in $OUT/$name.out"
    FAILED=1
  fi
}

if [ "$STAGE" = all ] || [ "$STAGE" = check ]; then
  step check bash scripts/a6000/check.sh
fi
if [ "$STAGE" = all ] || [ "$STAGE" = determinism ]; then
  step determinism "$PY" -u scripts/measure/session_determinism.py "$OUT/determinism"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = concurrency ]; then
  NGPU=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")
  [ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1
  say "concurrency on $NGPU GPU(s): the curve on card 0, then every card at once if there is more than one"
  step concurrency "$PY" -u scripts/measure/session1_concurrency.py "$OUT/concurrency" --gpus "$NGPU"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = recheck ]; then
  step recheck "$PY" -u scripts/measure/calibrate_s4.py runs/s4-recheck --frozen
fi

tar -cf results/session2.tar -C runs --exclude='session2/concurrency/*.npz' --exclude='s1cache-*' session2 s4-recheck
[ -f results/MACHINE_VERIFIED ] && tar -rf results/session2.tar results/MACHINE_VERIFIED
if [ "$FAILED" = 0 ]; then
  say "DONE. Send the single file results/session2.tar ($(du -h results/session2.tar | cut -f1))."
else
  say "DONE WITH PROBLEMS (see above). Send the single file results/session2.tar anyway: the logs are in it."
fi
