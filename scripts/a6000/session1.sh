#!/usr/bin/env bash
# Pre-run checklist items 4.1 to 4.3 on the A6000, before the long run. docs/increment1-prerun-checklist.md.
#
#   bash scripts/a6000/session1.sh               all three, in order. About 1 to 2 hours in total
#   bash scripts/a6000/session1.sh cache         4.1 only: why the compilation cache stopped writing. A few minutes
#   bash scripts/a6000/session1.sh determinism   4.2 only: one compiled program across 13 fresh processes. 20 to 40 min
#   bash scripts/a6000/session1.sh concurrency   4.3 only: how many runs at once, measured. 30 to 60 minutes
#
# Run inside tmux. Nothing in runs/sweep or runs/replay is changed; 4.2 only reads them. 4.1 may add up to three
# tiny entries to the compilation cache. Everything else goes into runs/session1, and at the end into the single
# file results/session1.tar, which is what to send back.
#
# It refuses to start while a sweep or a replay is running on this machine, because every timing would then be
# measured against our own load. The colleague's GPU job is expected and is measured, not refused: 4.3 records the
# background load before and after every round and says when a round could not be compared cleanly.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY="$ROOT/.venv/bin/python"
OUT=runs/session1
STAGE=${1:-all}
mkdir -p "$OUT" results
LOG="$OUT/session1.log"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  session1.sh  $*" | tee -a "$LOG"; }

case "$STAGE" in
all | cache | determinism | concurrency) ;;
*) echo "usage: bash scripts/a6000/session1.sh [all|cache|determinism|concurrency]"; exit 2 ;;
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
  say "STOPPED: a sweep, replay or calibration is running on this machine. Wait for it to finish, then run again."
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

say "rough durations: 4.1 a few minutes, 4.2 20 to 40 minutes, 4.3 30 to 60 minutes"
say "GPU right now: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null)"
say "(utilisation, memory used, memory total, temperature; anything already there is not ours)"

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

if [ "$STAGE" = all ] || [ "$STAGE" = cache ]; then
  step cache "$PY" -u scripts/measure/session1_cache.py "$OUT/cache"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = determinism ]; then
  if [ -d runs/sweep ] && [ -d runs/replay ]; then
    step determinism "$PY" -u scripts/measure/session1_determinism.py runs/sweep runs/replay "$OUT/determinism"
  else
    say "4.2 SKIPPED: it needs runs/sweep and runs/replay, and at least one is missing here."
    FAILED=1
  fi
fi
if [ "$STAGE" = all ] || [ "$STAGE" = concurrency ]; then
  step concurrency "$PY" -u scripts/measure/session1_concurrency.py "$OUT/concurrency"
fi

# 4.3's worker files hold whole populations (about 71 MB each) and are not needed back; its summaries, verdicts
# and logs are. 4.2's per-process files are small and come back with everything else.
tar -cf results/session1.tar -C runs --exclude='session1/concurrency/*.npz' --exclude='s1cache-*' session1
if [ "$FAILED" = 0 ]; then
  say "DONE. Send the single file results/session1.tar ($(du -h results/session1.tar | cut -f1))."
else
  say "DONE WITH PROBLEMS (see above). Send the single file results/session1.tar anyway: the logs are in it."
fi
