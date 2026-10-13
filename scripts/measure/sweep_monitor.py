"""Progress reporting and health checks for the sweep runner. D-062.

**Nothing here can change what the sweep computes, and that is the design constraint everything else bends
to.** Every function reads values the runner has already brought to the host: a generation's fitness vector,
its wall time, the process's memory, `nvidia-smi`. Nothing splits a random key, launches work on the GPU, or
writes anywhere the analysis reads. `test_sweep_progress.py` asserts the consequence directly: one seed run
with this module reporting at full verbosity is bitwise identical to the same seed run with it switched off.

**Why it exists.** The sweep is a multi-day job on a machine nobody will be watching closely, run by people
who can read a log but cannot change the code. Two things follow. Progress has to be visible at every scale
of the run, and anything out of place has to be NAMED in the log, with what it means and whether the run
handles it, rather than left for a reader to infer from numbers.

**The checks are the failure classes this project has actually met, each asserted on its cause where the
cause is observable, with the symptom kept as a second witness** (CLAUDE.md section 10):

    recompilation inside a seed   DEV-029's cause. JAX reports every trace through jax.monitoring; after a
                                  seed's first generations there should be none, so one is named at once
    host memory growth            DEV-029's symptom. Per seed, and across the run
    device memory growth          the same class on the GPU
    a generation far slower       a recompile the listener missed, contention, or a stall
    than its seed's median
    seed-to-seed slowdown         heat or another workload. Explained by the GPU snapshot beside it
    GPU heat, throttling, and     read from nvidia-smi once per seed
    other processes on the GPU
    non-finite fitness            a defect, never a result. Stops the run, because restarting repeats it
    a population scoring zero     reported, not stopped: harsh conditions can legitimately do this
    disk and host memory          stop cleanly before a write can fail, so a person can make room
    no progress at all            a watchdog thread names a stall and, past a limit, exits so the wrapper
                                  restarts the run from its last finished seed

**The thresholds below are operational, not scientific.** They decide when the log warns or the run stops,
never what is measured, so a wrong value costs a false alarm or a late one and cannot move a result.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

LEVELS = ("quiet", "normal", "verbose")

# Exit codes, read by scripts/a6000/run.sh to decide whether a restart can help.
EXIT_DONE = 0
EXIT_CRASH = 1  # an unexpected exception. Restartable: a resumed run loses at most the seed in progress
EXIT_FATAL = 2  # a precondition or an invariant failed. Restarting would repeat it, so a person must look
EXIT_DISK = 3  # disk nearly full. A person must make room; the run stopped before a write could fail
EXIT_HANG = 4  # the watchdog saw no progress. Restartable
EXIT_MEMORY = 5  # host memory nearly exhausted mid-run. Restartable, because the restart releases it
EXIT_INTERRUPT = 130  # Ctrl-C. The operator meant it, so the wrapper does not restart

# Thresholds. Operational, see the module docstring.
WARMUP_GENERATIONS = 2  # traces in a seed's first generations are its compilation, and expected
SPIKE_FACTOR = 5.0  # a generation this many times its seed's running median is named
SPIKE_MIN_SECONDS = 2.0  # but only if it also took this long, so jitter on a sub-second generation is quiet
DRIFT_FACTOR = 1.4  # a seed's median generation against the first seed of the same condition
RSS_SEED_GROWTH_MB = 400.0  # host memory added across one seed, from the condition's second seed on
RSS_RUN_GROWTH_MB = 2048.0  # host memory above the reading taken after the first finished condition
GPU_GROWTH_MB = 1024.0  # device memory in use above the first seed's reading
HOST_AVAILABLE_WARN_GB = 6.0
HOST_AVAILABLE_STOP_GB = 2.0
DISK_WARN_GB = 25.0
DISK_STOP_GB = 5.0
TEMP_WARN_C = 87.0
COLLAPSE_GENERATIONS = 100  # consecutive generations in which the best genome scored zero
STALL_WARN_SECONDS = 600.0
STALL_EXIT_SECONDS = 1800.0
STATUS_EVERY_SECONDS = 60.0
LIVE_EVERY_SECONDS = 0.5

# nvidia-smi's clock-event reasons that mean the GPU is being slowed against its will. A power cap under
# full load (0x4) is normal on any card and is reported in the snapshot without a warning.
THROTTLE_BITS = {
    0x8: "hardware slowdown",
    0x20: "software thermal slowdown",
    0x40: "hardware thermal slowdown",
    0x80: "hardware power brake",
}


class SweepStop(Exception):
    """Stop the run with a specific exit code and a message a person can act on."""

    exit_code = EXIT_CRASH

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class FatalStop(SweepStop):
    exit_code = EXIT_FATAL


class DiskStop(SweepStop):
    exit_code = EXIT_DISK


class MemoryStop(SweepStop):
    exit_code = EXIT_MEMORY


def exit_code_for(exc: BaseException) -> int:
    """Map whatever ended the run to the code the wrapper acts on.

    `SystemExit` with a message and `AssertionError` are the runner's own guards: the checkpoint schema, the
    verified commit, the naive_eval layout. Each is deterministic, so restarting repeats it, and they are
    fatal. Anything else is treated as a crash the restart can get past, because the runner resumes from its
    last finished seed and never repeats finished work.
    """
    if isinstance(exc, SweepStop):
        return exc.exit_code
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_INTERRUPT
    if isinstance(exc, SystemExit):
        if exc.code is None:
            return EXIT_DONE
        if isinstance(exc.code, int):
            return exc.code
        return EXIT_FATAL
    if isinstance(exc, AssertionError):
        return EXIT_FATAL
    if isinstance(exc, MemoryError):
        return EXIT_MEMORY
    return EXIT_CRASH


# ------------------------------------------------------------------------------------------- host readings --
def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def host_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return float("nan")


def disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1024.0**3
    except OSError:
        return float("nan")


def _smi(args: list[str]) -> list[list[str]] | None:
    try:
        out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")] for line in out.stdout.strip().splitlines() if line.strip()]


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def gpu_snapshot() -> dict:
    """One nvidia-smi reading. Never raises: an unreadable GPU is reported as unavailable, not as zero."""
    fields = (
        "name,driver_version,temperature.gpu,clocks.sm,power.draw,memory.used,memory.total,utilization.gpu"
    )
    rows = None
    for reasons in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        rows = _smi([f"--query-gpu={fields},{reasons}", "--format=csv,noheader,nounits"])
        if rows:
            break
    if not rows:
        return {"available": False}
    r = rows[0]
    snap = {
        "available": True,
        "name": r[0],
        "driver": r[1],
        "temp_c": _num(r[2]),
        "sm_mhz": _num(r[3]),
        "power_w": _num(r[4]),
        "mem_used_mb": _num(r[5]),
        "mem_total_mb": _num(r[6]),
        "util_pct": _num(r[7]),
    }
    try:
        snap["reasons"] = int(r[8], 16)
    except (ValueError, IndexError):
        snap["reasons"] = None
    apps = _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    snap["others"] = (
        [] if apps is None else [a for a in apps if a and a[0].isdigit() and int(a[0]) != os.getpid()]
    )
    return snap


def gpu_line(s: dict) -> str:
    if not s.get("available"):
        return "gpu unreadable"
    return f"gpu {s['temp_c']:.0f}C {s['sm_mhz']:.0f}MHz {s['power_w']:.0f}W util {s['util_pct']:.0f}%"


def fmt_duration(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d {h:02d}:{m:02d}"
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --------------------------------------------------------------------------------------------- the reporter --
class Monitor:
    """Progress lines, the live status line, the status file, the warnings log and every health check.

    The runner tells it where the run is (condition, seed, generation) and hands it the numbers it already
    has; the monitor decides what to print and what is out of place. One instance per process.
    """

    def __init__(
        self,
        level: str,
        out_dir: Path,
        plan: list[tuple[str, str, int]],
        n_seeds: int,
        n_generations: int,
        sample_every: int,
        *,
        log_path: Path | None = None,
        stream=None,
        clock=time.time,
        gpu=gpu_snapshot,
        rss=rss_mb,
        host_gb=host_available_gb,
        disk_gb=disk_free_gb,
        device_mb=None,
    ):
        assert level in LEVELS, f"--progress must be one of {LEVELS}, got {level!r}"
        self.level = level
        self.out_dir = Path(out_dir)
        self.stream = stream if stream is not None else sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.clock, self.gpu, self.rss, self.host_gb, self.disk_gb = clock, gpu, rss, host_gb, disk_gb
        self.device_mb = device_mb or (lambda: float("nan"))
        self.n_seeds, self.n_generations, self.sample_every = n_seeds, n_generations, sample_every
        self.log_path = Path(log_path) if log_path else self.out_dir / "sweep.log"
        self.warn_path = self.out_dir / "warnings.log"
        self.status_path = self.out_dir / "status.json"
        self._lock = threading.RLock()

        # plan: (arm, tag, seeds_remaining) for every condition, in plan order. Finished ones carry 0.
        self.plan = list(plan)
        self.total_gens = len(self.plan) * n_seeds * n_generations
        self.remaining_by_arm: dict[str, int] = {}
        for arm, _tag, left in self.plan:
            self.remaining_by_arm[arm] = self.remaining_by_arm.get(arm, 0) + left * n_generations
        self.done_gens = self.total_gens - sum(self.remaining_by_arm.values())
        self.rate_by_arm: dict[str, float] = {}

        self.started = self.clock()
        self.last_progress = self.started
        self._last_status = 0.0
        self._last_live = 0.0
        self.warnings_total = 0
        self.recent_warnings: deque = deque(maxlen=10)
        self._warned_once: set = set()

        self.cond_index = 0
        self.cond_tag = ""
        self.arm = ""
        self.seed = -1
        self.gen = -1
        self.state = "starting"
        self.exit_code = None

        # health state
        self.armed = False  # traces counted as unexpected only while a seed's loop is past its warm-up
        self.traces_in_seed = 0
        self.gen_times: list[float] = []
        self.zero_best_run = 0
        self.seed_rss_start = float("nan")
        self.first_seed_median = None
        self.device_ref_mb = None
        self.rss_ref_mb = None
        self.last_gpu: dict = {"available": False}
        self.last_fit: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------------------------------ output --
    def _stamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock()))

    def _write_log(self, path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(f"{self._stamp()}  {text}\n")
        except OSError:
            pass

    def line(self, text: str) -> None:
        """A permanent line: the terminal (clearing the live line first) and the log file."""
        with self._lock:
            if self.tty:
                self.stream.write("\r\x1b[2K")
            self.stream.write(text + "\n")
            self.stream.flush()
            self._write_log(self.log_path, text)

    def live(self, text: str, force: bool = False) -> None:
        """The in-place status line. Terminal only, and only when the terminal is a real one."""
        if not self.tty or self.level == "quiet":
            return
        now = self.clock()
        if not force and now - self._last_live < LIVE_EVERY_SECONDS:
            return
        self._last_live = now
        with self._lock:
            self.stream.write("\r\x1b[2K" + text[:200])
            self.stream.flush()

    def warn(self, code: str, text: str, once_key: str | None = None) -> None:
        """Name something out of place. Always printed, whatever the level, and kept in warnings.log."""
        if once_key is not None:
            if once_key in self._warned_once:
                return
            self._warned_once.add(once_key)
        if not self.cond_tag:
            where = "[setup]"
        elif self.seed < 0:
            where = f"[{self.cond_tag}]"
        else:
            where = f"[{self.cond_tag} seed {self.seed + 1}/{self.n_seeds} gen {self.gen + 1}/{self.n_generations}]"
        msg = f"WARNING {code} {where} {text}"
        with self._lock:
            self.warnings_total += 1
            self.recent_warnings.append(f"{self._stamp()} {code} {where} {text}")
            self.line(msg)
            self._write_log(self.warn_path, msg)
        self.write_status(force=True)

    # ------------------------------------------------------------------------------------------ status --
    def eta_seconds(self) -> float:
        total = 0.0
        known = [r for r in self.rate_by_arm.values() if math.isfinite(r)]
        fallback = statistics.fmean(known) if known else float("nan")
        for arm, left in self.remaining_by_arm.items():
            total += left * self.rate_by_arm.get(arm, fallback)
        return total

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "exit_code": self.exit_code,
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
            "updated": self._stamp(),
            "updated_epoch": self.clock(),
            "condition": f"{self.cond_index}/{len(self.plan)}",
            "condition_tag": self.cond_tag,
            "seed": f"{self.seed + 1}/{self.n_seeds}" if self.seed >= 0 else None,
            "generation": f"{self.gen + 1}/{self.n_generations}" if self.gen >= 0 else None,
            "generations_done": self.done_gens,
            "generations_total": self.total_gens,
            "percent_done": round(100.0 * self.done_gens / self.total_gens, 2) if self.total_gens else 100.0,
            "seconds_per_generation_by_arm": {k: round(v, 4) for k, v in self.rate_by_arm.items()},
            "eta": fmt_duration(self.eta_seconds()),
            "host_rss_mb": round(self.rss(), 1),
            "gpu": self.last_gpu,
            "warnings_total": self.warnings_total,
            "recent_warnings": list(self.recent_warnings),
        }

    def write_status(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._last_status < STATUS_EVERY_SECONDS:
            return
        self._last_status = now
        with self._lock:
            try:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.status_path.with_name("." + self.status_path.name)
                tmp.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
                tmp.replace(self.status_path)
            except OSError:
                pass

    # ----------------------------------------------------------------------------------- run structure --
    def run_start(self, header: list[str]) -> None:
        self.state = "running"
        for h in header:
            self.line(h)
        done_conditions = sum(1 for _a, _t, left in self.plan if left == 0)
        after = (
            " Nothing is left to run."
            if self.done_gens == self.total_gens
            else " The time remaining appears once the first generations are measured."
        )
        self.line(
            f"plan: {len(self.plan)} conditions x {self.n_seeds} seeds x {self.n_generations} generations; "
            f"{done_conditions} conditions already complete, {100.0 * self.done_gens / max(self.total_gens, 1):.1f}"
            f" percent of the work done.{after}"
        )
        self.write_status(force=True)

    def condition_start(self, index: int, arm: str, tag: str, seeds_done: int) -> None:
        self.cond_index, self.arm, self.cond_tag = index, arm, tag
        self.seed, self.gen = -1, -1
        self.first_seed_median = None
        self.check_resources()
        self.last_gpu = self.gpu()
        self.check_gpu(self.last_gpu)
        resumed = f", resuming with {seeds_done} of {self.n_seeds} seeds already done" if seeds_done else ""
        if self.level != "quiet":
            self.line(f"[{index:3d}/{len(self.plan)}] {tag} starting{resumed}")
        self.write_status(force=True)

    def seed_start(self, seed: int) -> None:
        self.seed, self.gen = seed, -1
        self.armed = False
        self.traces_in_seed = 0
        self.gen_times = []
        self.zero_best_run = 0
        self.seed_rss_start = self.rss()
        self.last_progress = self.clock()
        self._seed_t0 = self.clock()

    def seed_skipped(self, seed: int) -> None:
        if self.level == "verbose":
            self.line(f"  seed {seed + 1}/{self.n_seeds} already done, reused from its partial checkpoint")

    # The runner's one_seed calls these two. They must only read what they are given.
    def generation(self, gen: int, seconds: float, fit, acc_row=None) -> None:
        self.gen = gen
        now = self.clock()
        self.last_progress = now
        self.done_gens += 1
        self.remaining_by_arm[self.arm] = max(0, self.remaining_by_arm.get(self.arm, 0) - 1)
        if gen >= 1:  # generation 0 carries the compilation, so it is kept out of the rate
            prev = self.rate_by_arm.get(self.arm)
            self.rate_by_arm[self.arm] = seconds if prev is None else 0.98 * prev + 0.02 * seconds
            self.gen_times.append(seconds)

        mean, best = float(fit.mean()), float(fit.max())
        zero = float((fit == 0).mean())
        self.last_fit = (mean, best, zero)
        self.check_generation(gen, seconds, mean, best)
        if gen >= WARMUP_GENERATIONS - 1:
            self.armed = True

        sample = acc_row is not None
        if self.level == "verbose" or (self.level == "normal" and sample):
            extra = ""
            if sample and len(acc_row) >= 3:
                extra = f"  predated {acc_row[-3]:.3f} starved {acc_row[-2]:.3f} survived {acc_row[-1]:.3f}"
            self.line(
                f"    gen {gen + 1:3d}/{self.n_generations}  {seconds:6.3f} s  fit {mean:.4f} best {best:.4f} "
                f"zero {100 * zero:4.1f}%{extra}"
            )
        self.live(self._live_text(seconds))
        self.write_status()

    def _live_text(self, seconds: float) -> str:
        mean, best, zero = self.last_fit or (float("nan"),) * 3
        left_seed = (self.n_generations - self.gen - 1) * self.rate_by_arm.get(self.arm, seconds)
        return (
            f"[cond {self.cond_index}/{len(self.plan)} | seed {self.seed + 1}/{self.n_seeds} | gen "
            f"{self.gen + 1}/{self.n_generations}]  {seconds:.3f} s/gen  fit {mean:.4f} best {best:.4f} "
            f"zero {100 * zero:.0f}%  eta seed {fmt_duration(left_seed)}  sweep {fmt_duration(self.eta_seconds())}"
        )

    def loop_end(self) -> None:
        """The seed's generations are done. Traces after this point (the null arm's naive_eval) are expected."""
        self.armed = False

    def seed_end(self, seed: int, final_mean: float, final_best: float) -> None:
        wall = self.clock() - self._seed_t0
        med = statistics.median(self.gen_times) if self.gen_times else float("nan")
        rss_now = self.rss()
        dev = self.device_mb()
        self.last_gpu = self.gpu()
        self.check_seed(seed, med, rss_now, dev, self.last_gpu)
        if self.level != "quiet":
            self.line(
                f"  seed {seed + 1}/{self.n_seeds} done in {fmt_duration(wall)}  final {final_mean:.4f} best "
                f"{final_best:.4f}  {med:.3f} s/gen  ram {rss_now / 1024:.2f} GB  gpu mem {dev:.0f} MB  "
                f"{gpu_line(self.last_gpu)}"
            )
        self.write_status(force=True)

    def condition_end(self, tag: str, wall: float, final_mean: float, sem: float) -> None:
        rss_now = self.rss()
        if self.rss_ref_mb is None:
            self.rss_ref_mb = rss_now
        elif math.isfinite(rss_now) and rss_now - self.rss_ref_mb > RSS_RUN_GROWTH_MB:
            self.warn(
                "MEMORY-RUN",
                f"host memory is {rss_now - self.rss_ref_mb:.0f} MB above its reading after the first condition "
                f"({self.rss_ref_mb:.0f} to {rss_now:.0f} MB). A slow leak across conditions. The run continues; "
                f"it stops and restarts itself if free memory falls below {HOST_AVAILABLE_STOP_GB} GB.",
                once_key=f"rssrun{int((rss_now - self.rss_ref_mb) // RSS_RUN_GROWTH_MB)}",
            )
        pct = 100.0 * self.done_gens / max(self.total_gens, 1)
        self.line(
            f"[{self.cond_index:3d}/{len(self.plan)}] {tag:<44s} {wall / 60:5.1f} min  fit {final_mean:.3f} +/- "
            f"{sem:.3f}  ram {rss_now:.0f}MB  done {pct:5.1f}%  elapsed {fmt_duration(self.clock() - self.started)}"
            f"  remaining ~{fmt_duration(self.eta_seconds())}"
        )
        self.write_status(force=True)

    def run_end(self, code: int, reason: str = "") -> None:
        self.exit_code = code
        self.state = "complete" if code == EXIT_DONE else "stopped"
        msg = f"run {self.state}, exit code {code}" + (f": {reason}" if reason else "")
        self.line(msg)
        self.write_status(force=True)

    # ---------------------------------------------------------------------------------------- checks --
    def on_trace(self, *_args, **_kwargs) -> None:
        """jax.monitoring listener for jaxpr traces. Unexpected only inside a seed's loop past its warm-up."""
        if not self.armed:
            return
        self.traces_in_seed += 1
        if self.traces_in_seed == 1:
            self.warn(
                "RECOMPILE",
                "JAX traced a function in the middle of a seed. Every generation should reuse the same compiled "
                "program; a trace here is the cause of DEV-029, where one fresh program per generation leaked "
                "about 31 MB each and killed the run. Watch the MEMORY warnings that would follow.",
            )

    def check_generation(self, gen: int, seconds: float, mean: float, best: float) -> None:
        if not (math.isfinite(mean) and math.isfinite(best)):
            raise FatalStop(
                f"non-finite fitness at {self.cond_tag} seed {self.seed} generation {gen}: mean {mean}, best "
                f"{best}. Fitness counts collected targets, so this is a defect, not a result, and restarting "
                "would reproduce it exactly. Nothing after this point was saved."
            )
        if gen >= WARMUP_GENERATIONS and len(self.gen_times) >= 5:
            med = statistics.median(self.gen_times[:-1])
            if seconds > SPIKE_FACTOR * med and seconds > SPIKE_MIN_SECONDS:
                self.warn(
                    "SLOW-GENERATION",
                    f"one generation took {seconds:.2f} s against this seed's median of {med:.3f} s. Causes, most "
                    "likely first: another program used the GPU, the GPU throttled, or a recompile (a RECOMPILE "
                    "warning would name that). Harmless if it does not repeat.",
                    once_key=f"spike{self.cond_tag}{self.seed}",
                )
        self.zero_best_run = self.zero_best_run + 1 if best == 0 else 0
        if self.zero_best_run == COLLAPSE_GENERATIONS:
            self.warn(
                "ZERO-POPULATION",
                f"the best genome has scored zero for {COLLAPSE_GENERATIONS} generations in a row. This can be a "
                "genuine result in a harsh condition, so the run continues; it is named so it is looked at.",
            )

    def check_seed(self, seed: int, med: float, rss_now: float, dev_mb: float, gpu: dict) -> None:
        if self.first_seed_median is None:
            self.first_seed_median = med
        elif math.isfinite(med) and med > DRIFT_FACTOR * self.first_seed_median:
            self.warn(
                "SLOWDOWN",
                f"this seed ran at {med:.3f} s per generation against {self.first_seed_median:.3f} s for the "
                f"first seed of the same condition, {med / self.first_seed_median:.2f} times slower. Same work, so "
                f"the machine changed: {gpu_line(gpu)}. Look for heat or another program on the GPU.",
                once_key=f"drift{self.cond_tag}",
            )
        grown = rss_now - self.seed_rss_start
        if math.isfinite(grown) and grown > RSS_SEED_GROWTH_MB and self.seed > 0:
            self.warn(
                "MEMORY-SEED",
                f"host memory grew {grown:.0f} MB during one seed ({self.seed_rss_start:.0f} to {rss_now:.0f} MB). "
                "A seed after the first should not hold on to memory. If this repeats, the run is leaking.",
            )
        if self.device_ref_mb is None and math.isfinite(dev_mb):
            self.device_ref_mb = dev_mb
        elif (
            self.device_ref_mb is not None
            and math.isfinite(dev_mb)
            and dev_mb - self.device_ref_mb > GPU_GROWTH_MB
        ):
            self.warn(
                "GPU-MEMORY",
                f"GPU memory in use is {dev_mb - self.device_ref_mb:.0f} MB above the first seed's reading "
                f"({self.device_ref_mb:.0f} to {dev_mb:.0f} MB). The card has room to spare, so the run "
                "continues, but steady growth is a leak.",
                once_key=f"gpumem{int((dev_mb - self.device_ref_mb) // GPU_GROWTH_MB)}",
            )
        self.check_gpu(gpu)
        self.check_resources()

    def check_gpu(self, gpu: dict) -> None:
        if not gpu.get("available"):
            self.warn(
                "GPU-UNREADABLE",
                "nvidia-smi gave no reading. The run continues; heat cannot be watched.",
                once_key="smi",
            )
            return
        if math.isfinite(gpu["temp_c"]) and gpu["temp_c"] >= TEMP_WARN_C:
            self.warn(
                "GPU-HOT",
                f"GPU at {gpu['temp_c']:.0f} C. Check the case fans and airflow; the card slows "
                "itself before it is damaged, so the cost is time.",
                once_key=f"hot{self.cond_tag}",
            )
        reasons = gpu.get("reasons")
        if reasons:
            named = [v for bit, v in THROTTLE_BITS.items() if reasons & bit]
            if named:
                self.warn(
                    "GPU-THROTTLE",
                    f"the GPU reports {', '.join(named)} (reasons {reasons:#x}). The run "
                    "continues more slowly.",
                    once_key=f"thr{self.cond_tag}{reasons}",
                )
        for other in gpu.get("others", []):
            self.warn(
                "GPU-SHARED",
                f"another process is using the GPU: pid {other[0]} {other[1] if len(other) > 1 else ''}. The "
                "sweep's timing assumes it has the card to itself; results are unaffected, but it runs slower.",
                once_key=f"other{other[0]}",
            )

    def check_resources(self) -> None:
        free = self.disk_gb(self.out_dir)
        if math.isfinite(free):
            if free < DISK_STOP_GB:
                raise DiskStop(
                    f"only {free:.1f} GB free where the checkpoints are written ({self.out_dir}). A condition "
                    f"needs about 1.5 GB while it runs. The run stopped before a write could fail and nothing is "
                    "lost: free some space and run the same command again."
                )
            if free < DISK_WARN_GB:
                self.warn(
                    "DISK-LOW",
                    f"{free:.1f} GB free for checkpoints. The run stops cleanly at "
                    f"{DISK_STOP_GB} GB; make room before then.",
                    once_key=f"disk{int(free)}",
                )
        avail = self.host_gb()
        if math.isfinite(avail):
            if avail < HOST_AVAILABLE_STOP_GB:
                raise MemoryStop(
                    f"only {avail:.1f} GB of host memory available. The run stops so the wrapper can restart it, "
                    "which releases everything this process holds; it resumes from the last finished seed."
                )
            if avail < HOST_AVAILABLE_WARN_GB:
                self.warn(
                    "HOST-MEMORY",
                    f"{avail:.1f} GB of host memory available. Close other programs.",
                    once_key=f"host{int(avail)}",
                )


# --------------------------------------------------------------------------------------------- watchdog --
def start_watchdog(
    mon: Monitor,
    warn_after: float = STALL_WARN_SECONDS,
    exit_after: float = STALL_EXIT_SECONDS,
    poll: float = 30.0,
    exit_fn=os._exit,
) -> threading.Thread:
    """A daemon thread that names a stall, and past `exit_after` ends the process so the wrapper restarts it.

    It cannot see a hang that holds Python's interpreter lock, so `run.sh` also watches the status file's age
    from outside the process. Between the two, a stuck run does not sit stuck until someone looks.
    """
    state = {"warned_for": None}

    def loop():
        while True:
            time.sleep(poll)
            if mon.state != "running":
                continue
            idle = mon.clock() - mon.last_progress
            if idle > exit_after:
                mon.warn(
                    "STALL",
                    f"no progress for {fmt_duration(idle)}. Exiting so the wrapper restarts the run "
                    "from its last finished seed.",
                )
                mon.run_end(EXIT_HANG, "stalled")
                exit_fn(EXIT_HANG)
                return
            if idle > warn_after and state["warned_for"] != mon.last_progress:
                state["warned_for"] = mon.last_progress
                mon.warn(
                    "STALL",
                    f"no generation has finished for {fmt_duration(idle)}. A compile takes under two "
                    f"minutes, so this is unusual. The run exits and restarts itself at "
                    f"{fmt_duration(exit_after)}.",
                )

    t = threading.Thread(target=loop, name="sweep-watchdog", daemon=True)
    t.start()
    return t


def acquire_run_lock(out_dir: Path):
    """One sweep per output directory. Two at once would race on the same partial checkpoints.

    An OS advisory lock, released by the kernel when the process dies however it dies, so a crash never leaves
    a stale lock behind. Returns the open file, which must stay referenced for the lock to hold.
    """
    import fcntl

    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "sweep.lock", "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise FatalStop(
            f"another sweep is already running in {out_dir}. Only one may run at a time; two would overwrite each "
            "other's checkpoints. Find it with `ps aux | grep sweep.py` or reattach its tmux session."
        ) from None
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh
