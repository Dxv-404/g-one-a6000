#!/usr/bin/env bash
# Run the baselines and the sweep on the A6000, unattended. D-061, D-062.
#
#   bash scripts/a6000/run.sh baselines    the two baseline conditions, then gates S5 and S6. 1 to 2 hours
#   bash scripts/a6000/run.sh sweep        the sweep. Needs the baselines. Roughly 2 to 3 days
#   bash scripts/a6000/run.sh all          both, one after the other
#   bash scripts/a6000/run.sh all --smoke  the same control flow at toy size, into runs/smoke-run
#
# Run inside tmux. Safe to run again at any time with the same arguments: finished seeds and conditions are
# kept and never repeated, so after a crash, a reboot or a power cut the same command simply continues.
#
# What it does on its own, so nobody has to:
#   restarts the sweep after a crash, a hang or exhausted memory, up to MAX_RESTARTS times
#   stops for good, with a message, on anything a restart cannot fix: a failed check, a full disk, Ctrl-C
#   kills and restarts a run whose status.json has not changed for STALE_MINUTES, from outside the process,
#     which catches a hang the run's own watchdog cannot see
#   refuses to start on a machine that has not passed check.sh, or if any shipped file has been edited
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
STAGE=${1:-}
SMOKE=0
[ "${2:-}" = "--smoke" ] && SMOKE=1
SHA=$(cat VERIFIED_COMMIT)  # the run code is byte-identical to this commit, which passed the full differential
PY="$ROOT/.venv/bin/python"
MAX_RESTARTS=8
STALE_MINUTES=45
OUT=runs/sweep
EXTRA=()
if [ "$SMOKE" = 1 ]; then
  OUT=runs/smoke-run
  EXTRA=(--smoke)
fi
mkdir -p "$OUT" results
RUNLOG="$OUT/run.log"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  run.sh  $*" | tee -a "$RUNLOG"; }

usage() {
  echo "usage: bash scripts/a6000/run.sh baselines|sweep|all [--smoke]"
  exit 2
}

preflight() {
  [ -x "$PY" ] || { say "no environment at .venv: run  bash scripts/a6000/check.sh  first."; exit 2; }
  if ! sha256sum -c --quiet COPY-MANIFEST.sha256 >/dev/null 2>&1; then
    say "STOPPED: a shipped file has been changed. The run must use the code exactly as shipped."
    sha256sum -c --quiet COPY-MANIFEST.sha256 2>&1 | head -5 | tee -a "$RUNLOG"
    say "Undo the change with  git checkout -- .  and run this command again."
    exit 2
  fi
  if [ "$SMOKE" = 0 ] && [ ! -f results/MACHINE_VERIFIED ]; then
    say "STOPPED: this machine has not passed the check. Run  bash scripts/a6000/check.sh  and let it pass."
    exit 2
  fi
  if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "${ALLOW_NO_TMUX:-0}" != 1 ]; then
    say "WARNING: not inside tmux. If this terminal or its VS Code window closes, the run dies with it."
    say "         Start  tmux new -s gone  and run this command inside it. Continuing in 20 seconds."
    sleep 20
  fi
}

# supervise <description> <extra sweep arguments...>
supervise() {
  local desc=$1
  shift
  local attempt=0 pid rc launched age mtime
  while :; do
    attempt=$((attempt + 1))
    say "starting $desc (attempt $attempt of $MAX_RESTARTS)"
    "$PY" -u scripts/measure/sweep.py "$OUT" --verified-commit "$SHA" ${EXTRA[@]+"${EXTRA[@]}"} "$@" &
    pid=$!
    launched=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      mtime=$launched
      [ -f "$OUT/status.json" ] && mtime=$(stat -c %Y "$OUT/status.json")
      [ "$mtime" -lt "$launched" ] && mtime=$launched
      age=$(($(date +%s) - mtime))
      if [ "$age" -gt $((STALE_MINUTES * 60)) ] && kill -0 "$pid" 2>/dev/null; then
        say "status.json has not changed for $((age / 60)) minutes. Stopping the stuck process so it can restart."
        kill -TERM "$pid" 2>/dev/null
        sleep 60
        kill -KILL "$pid" 2>/dev/null
      fi
    done
    wait "$pid"
    rc=$?
    case $rc in
    0)
      say "$desc finished"
      return 0
      ;;
    2)
      say "STOPPED (exit 2): a check failed that a restart cannot fix. The reason is the last line of $OUT/sweep.log."
      exit 2
      ;;
    3)
      say "STOPPED (exit 3): the disk is nearly full. Free space, then run the same command again. Nothing is lost."
      exit 3
      ;;
    130)
      say "stopped by Ctrl-C (exit 130). Run the same command again to continue."
      exit 130
      ;;
    esac
    if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
      say "STOPPED: $attempt attempts in a row ended in a crash (last exit $rc). Send $OUT/sweep.log and $RUNLOG."
      exit 1
    fi
    say "ended with exit $rc, which a restart can get past. Restarting in 60 seconds; finished seeds are kept."
    sleep 60
  done
}

baselines() {
  if [ "$SMOKE" = 1 ]; then
    supervise "the smoke plan"
    return 0
  fi
  supervise "the treatment baseline" --arms treatment --limit 1
  supervise "the null baseline" --arms null --limit 1
  say "gate S5: is the null well behaved? Required."
  if ! "$PY" -m pytest tests/scientific/test_s5_s6_null.py -k s5 -q >results/s5.log 2>&1; then
    tail -n 15 results/s5.log | tee -a "$RUNLOG"
    say "STOPPED: gate S5 failed on this machine's null baseline. Send results/s5.log and $OUT/sweep.log."
    exit 2
  fi
  tail -n 1 results/s5.log | tee -a "$RUNLOG"
  say "gate S6: statistical power at ten seeds. Recorded, not required: it is expected to FAIL under its"
  say "         conservative assumption, and the seed count is the project owner's decision."
  "$PY" -m pytest tests/scientific/test_s5_s6_null.py -k s6 -q >results/s6.log 2>&1
  tail -n 1 results/s6.log | tee -a "$RUNLOG"
  say "the paired power analysis on this machine's baselines"
  "$PY" -u scripts/measure/pairing_power.py results/pairing_power.npz >results/pairing_power.log 2>&1
  tail -n 12 results/pairing_power.log | tee -a "$RUNLOG"
  {
    echo "baselines done $(date '+%Y-%m-%d %H:%M:%S')"
    echo "S5: $(tail -n 1 results/s5.log)"
    echo "S6: $(tail -n 1 results/s6.log)"
  } >results/BASELINES_DONE
  say "baselines done. Send results/s5.log, results/s6.log and results/pairing_power.log to the project team."
}

sweep() {
  if [ "$SMOKE" = 0 ] && [ ! -f results/BASELINES_DONE ]; then
    say "STOPPED: run  bash scripts/a6000/run.sh baselines  first."
    exit 2
  fi
  local free
  free=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
  [ "$free" -lt 90 ] && say "WARNING: ${free} GB free; the sweep needs about 80 GB. It stops cleanly if space runs out."
  supervise "the sweep"
  (cd "$OUT" && sha256sum ./*.npz manifest.jsonl >"$ROOT/results/sweep-checksums.sha256")
  echo "sweep done $(date '+%Y-%m-%d %H:%M:%S')" >results/SWEEP_DONE
  say "SWEEP COMPLETE. Send the folder $OUT and the folder results. Checksums: results/sweep-checksums.sha256"
}

case "$STAGE" in
baselines) preflight; baselines ;;
sweep) preflight; sweep ;;
all) preflight; baselines; sweep ;;
*) usage ;;
esac
