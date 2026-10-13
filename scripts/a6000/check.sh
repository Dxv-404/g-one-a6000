#!/usr/bin/env bash
# Machine check for the A6000 run. D-061, D-062.
#
#   bash scripts/a6000/check.sh
#
# Run from anywhere inside the repository, inside tmux. Takes roughly 45 to 75 minutes. It sets up the
# environment from the lock file, proves the GPU is used, runs every test that concerns the code and the
# hardware, runs the whole sweep path at toy size, and measures this machine's speed. It ends with
#
#   MACHINE CHECK PASSED    and writes results/MACHINE_VERIFIED, which run.sh requires, or
#   MACHINE CHECK FAILED    and names the step. Nothing is written that run.sh would accept.
#
# Every step's full output is kept in results/check-<time>/. Send that folder if anything fails.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
TS=$(date +%Y%m%d-%H%M%S)
OUT="results/check-$TS"
mkdir -p "$OUT"
SHA=$(cat VERIFIED_COMMIT 2>/dev/null || echo unknown)  # the run code is byte-identical to this verified commit
PY="$ROOT/.venv/bin/python"
SUMMARY=()
FAILED=""

say() { echo "$(date '+%H:%M:%S')  $*"; }

# step <required: yes|no> <name> <log file> <command...>
step() {
  local required=$1 name=$2 log=$3
  shift 3
  local t0 rc dt
  t0=$(date +%s)
  echo
  say "=== $name"
  say "    full output: $OUT/$log"
  "$@" >"$OUT/$log" 2>&1
  rc=$?
  dt=$(($(date +%s) - t0))
  tail -n 4 "$OUT/$log" | sed 's/^/      /'
  if [ "$rc" -eq 0 ]; then
    SUMMARY+=("$(printf 'PASS  %5ds  %s' "$dt" "$name")")
  elif [ "$required" = yes ]; then
    SUMMARY+=("$(printf 'FAIL  %5ds  %s  (exit %s)' "$dt" "$name" "$rc")")
    FAILED="$name"
  else
    SUMMARY+=("$(printf 'NOTE  %5ds  %s  (exit %s, measurement only)' "$dt" "$name" "$rc")")
  fi
  return "$rc"
}

finish() {
  echo
  echo "======================================================================================"
  for s in "${SUMMARY[@]}"; do echo "  $s"; done
  echo "======================================================================================"
  if [ -n "$FAILED" ]; then
    echo "MACHINE CHECK FAILED at: $FAILED"
    echo "Do not start the run. Send the folder $OUT to the project team."
    rm -f results/MACHINE_VERIFIED
    exit 1
  fi
  {
    echo "machine check passed $(date '+%Y-%m-%d %H:%M:%S')"
    echo "source commit $(cat SOURCE_COMMIT 2>/dev/null)   verified commit $SHA"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "results $OUT"
  } >results/MACHINE_VERIFIED
  echo "MACHINE CHECK PASSED. results/MACHINE_VERIFIED written."
  echo "Next:  bash scripts/a6000/run.sh baselines"
  exit 0
}

# The eighteen tests that check the full repository's own records (its run registry, its unswept-parameter
# list, its repository map, its git history) rather than the code or the hardware. They pass in the full
# repository and are deselected here, where those records are not shipped.
DESELECT_ENG=(
  --deselect tests/engineering/test_repository_map.py
  --deselect tests/engineering/test_design_audit.py::test_the_known_unrooted_parameters_are_on_the_unswept_list
  --deselect tests/engineering/test_design_audit.py::test_every_unswept_payoff_lever_is_declared_on_the_unswept_list
  --deselect tests/engineering/test_design_audit.py::test_a_measured_root_names_a_run_that_exists
  --deselect tests/engineering/test_design_audit.py::test_the_measured_root_check_can_fail
  --deselect tests/engineering/test_environment_capture.py::test_the_dirty_flag_actually_responds
)
DESELECT_SCI=(
  --deselect tests/scientific/test_differential_binding.py::test_every_bound_run_has_a_recorded_differential_pass
  --deselect tests/scientific/test_preregistration.py::test_every_preregistered_file_exists_and_has_a_commit
  --deselect tests/scientific/test_preregistration.py::test_every_registered_run_declares_its_phase_unambiguously
  --deselect tests/scientific/test_preregistration.py::test_no_phase_b_run_predates_its_pre_registration
  --deselect tests/scientific/test_preregistration.py::test_the_marker_convention_is_documented
  # S5 and S6 read the null baseline, which run.sh produces on this machine. run.sh runs them then.
  --deselect tests/scientific/test_s5_s6_null.py
)

machine_facts() {
  echo "date        $(date)"
  echo "host        $(hostname)   $(uname -srm)"
  echo "folder      $ROOT"
  echo "commit      $SHA"
  echo "cpu         $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  echo "memory      $(free -g | awk '/^Mem:/{print $2 " GB total, " $7 " GB available"}')"
  echo "disk free   $(df -BG --output=avail . | tail -1 | tr -d ' ') where the results go"
  echo "tmux        $(command -v tmux || echo 'NOT INSTALLED: the run must be started inside tmux or with nohup')"
  echo "uv          $(uv --version 2>/dev/null || echo 'NOT INSTALLED')"
  echo
  nvidia-smi
}

need_space() {
  local free
  free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  echo "free space: ${free} GB. The sweep needs about 80 GB for its checkpoints; 100 GB is required to start."
  [ "$free" -ge 100 ]
}

on_linux_filesystem() {
  case "$ROOT" in
  /mnt/*)
    echo "this folder is under $ROOT, a Windows drive seen from WSL. Small-file writes there are hundreds"
    echo "of times slower and JAX's cache lives on them. Clone the repository into the Linux home folder."
    return 1
    ;;
  esac
  echo "on a Linux filesystem: $ROOT"
}

smoke_twice() {
  rm -rf runs/smoke-check
  "$PY" -u scripts/measure/sweep.py runs/smoke-check --smoke --verified-commit "$SHA" || return 1
  echo "---- second run: everything must already be complete"
  "$PY" -u scripts/measure/sweep.py runs/smoke-check --smoke --verified-commit "$SHA" || return 1
  "$PY" - <<'EOF' || return 1
import json, pathlib
d = json.loads(pathlib.Path("runs/smoke-check/status.json").read_text())
assert d["state"] == "complete" and d["exit_code"] == 0, d
assert d["generations_done"] == d["generations_total"], d
print("status.json: complete,", d["generations_done"], "of", d["generations_total"], "generations")
EOF
  rm -rf runs/smoke-check
}

say "machine check for the A6000 run, commit $SHA. Results in $OUT"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver. Stop here."; exit 1; }
command -v uv >/dev/null || { echo "uv not found. Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

step yes "machine facts" 00-machine.log machine_facts || finish
step yes "folder on a Linux filesystem" 01-filesystem.log on_linux_filesystem || finish
step yes "every file as shipped (checksums)" 02-manifest.log sha256sum -c --quiet COPY-MANIFEST.sha256 || finish
step yes "disk space" 03-disk.log need_space || finish
step yes "environment from the lock file (first time downloads about 3 GB)" 04-uv-sync.log uv sync --frozen || finish
step yes "the GPU is used, not the CPU (gate E0)" 05-e0.log "$PY" -m pytest tests/engineering/test_e0_setup.py -q || finish
step yes "engineering tests" 06-engineering.log "$PY" -m pytest tests/engineering -q "${DESELECT_ENG[@]}" || finish
step yes "differential tests against the reference implementation" 07-differential.log "$PY" -m pytest tests/differential -q || finish
step yes "scientific tests" 08-scientific.log "$PY" -m pytest tests/scientific -q "${DESELECT_SCI[@]}" || finish
step yes "the sweep at toy size, twice, to prove resume" 09-smoke.log smoke_twice || finish
step no "speed under sustained load (15 minutes)" 10-sustained.log "$PY" -u scripts/measure/sustained_load.py --minutes 15
step no "speed against batch size" 11-batch.log "$PY" -u scripts/measure/batch_scaling.py
step no "where a tick's time goes" 12-profile.log "$PY" -u scripts/measure/tick_profile.py
finish
