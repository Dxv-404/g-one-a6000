#!/usr/bin/env bash
# Replay the sweep's final populations on the A6000 and measure what the sweep did not keep. Run 48.
#
#   bash scripts/a6000/replay.sh              all 148 jobs, into runs/replay
#   bash scripts/a6000/replay.sh --plan-only  list the jobs and stop
#
# Run inside tmux. Safe to run again at any time: finished jobs are kept and skipped.
#
# It needs the sweep's checkpoints in runs/sweep, which hold the final populations. It does NOT need a fresh
# check.sh pass, and that is deliberate: before measuring anything new, every job reproduces the sweep's own
# stored numbers bit for bit, and stops if it cannot. That proves this machine and this code reproduce the
# sweep more directly than any check could. It still refuses edited code and a missing GPU.
#
# When it finishes it writes results/replay.tar, which is the only thing that needs copying back.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PY="$ROOT/.venv/bin/python"
OUT=runs/replay
SWEEP=runs/sweep
MAX_RESTARTS=4
mkdir -p "$OUT" results
LOG="$OUT/run.log"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  replay.sh  $*" | tee -a "$LOG"; }

. scripts/a6000/env.sh
scrub_environment >"$OUT/environment.txt"
while read -r line; do say "$line"; done <"$OUT/environment.txt"

[ -x "$PY" ] || { say "no environment at .venv: run  bash scripts/a6000/check.sh  once first."; exit 2; }
if ! sha256sum -c --quiet COPY-MANIFEST.sha256 >/dev/null 2>&1; then
  say "STOPPED: a shipped file has been changed. Undo it with  git checkout -- .  and run again."
  exit 2
fi
if [ "${1:-}" = "--plan-only" ]; then
  "$PY" scripts/measure/replay_final.py "$SWEEP" "$OUT" --plan-only
  exit $?
fi
[ -d "$SWEEP" ] || { say "STOPPED: no $SWEEP. The replay reads the sweep's checkpoints from there."; exit 2; }
if ! gpu_probe "$PY" >"$OUT/gpu_probe.txt" 2>&1; then
  tail -n 3 "$OUT/gpu_probe.txt" | tee -a "$LOG"
  say "STOPPED: JAX cannot use the GPU from this terminal. Check that nvidia-smi works, then run again."
  exit 2
fi
say "$(tail -n 1 "$OUT/gpu_probe.txt")"
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
  say "WARNING: not inside tmux. If this terminal closes, the replay stops (it continues when run again)."
  sleep 10
fi

attempt=0
while :; do
  attempt=$((attempt + 1))
  say "starting the replay (attempt $attempt of $MAX_RESTARTS)"
  "$PY" -u scripts/measure/replay_final.py "$SWEEP" "$OUT" &
  pid=$!
  trap 'kill -TERM "$pid" 2>/dev/null; say "stopped by Ctrl-C. Finished jobs are kept; run again to continue."; exit 130' INT TERM
  wait "$pid"
  rc=$?
  trap - INT TERM
  case "$rc" in
  0) break ;;
  3) say "STOPPED on purpose (exit 3): the reason is printed just above. Do not simply run it again; report it."
     exit 3 ;;
  *) if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
       say "STOPPED: the replay failed $attempt times (last exit $rc). Send runs/replay/run.log and replay.log."
       exit "$rc"
     fi
     say "the replay exited with $rc; restarting in 60 seconds (finished jobs are kept)"
     sleep 60 ;;
  esac
done

tar -cf results/replay.tar -C runs replay
say "DONE. Copy back the single file results/replay.tar ($(du -h results/replay.tar | cut -f1))."
