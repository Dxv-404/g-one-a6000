"""Shared helpers for the A6000 session 1 checks (pre-run checklist items 4.1 to 4.3). Host-side only.

Nothing here imports JAX or the project, so a driver can sample the machine before any JAX process exists, and
the pure functions are unit-tested in `tests/engineering/test_session1_checks.py`.

**Why background load is sampled at all.** The A6000 machine is WSL2 on a Windows desktop. On 2026-11-06
`nvidia-smi` inside WSL reported the GPU 55 percent utilised with 9,395 MiB in use while it listed no compute
processes and the WSL load average was 0.16. The Windows side named the cause: a colleague's nnU-Net training job
(a Windows Python process, compute type) holding about 9.4 GB and running intermittently, plus ordinary desktop
programs, on a GPU in WDDM mode that also drives the display. The owner's planning assumes that job KEEPS RUNNING
through the long run. So every throughput figure here is measured beside a foreign workload that changes on its
own schedule, and every round records the background load before AND after it, with none of our processes running.

A round is CLEAN only if both of its background readings are within the margins below of the reference, which is
the first background reading of the session. A round that is not clean still reports bit-identity, because that
verdict does not depend on load, but its throughput is excluded from the stop rule and the recommendation.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import shutil
import statistics
import subprocess
import threading
import time

#: A background reading is NOT CLEAN when its mean utilisation differs from the reference by MORE than this many
#: percentage points, or its mean memory by MORE than this many GiB. Both chosen, not derived: 10 points is
#: larger than the second-to-second jitter of an idle GPU and smaller than a training job switching on; 1 GiB is
#: about a tenth of the foreign job's footprint seen on 2026-11-06. Strictness thresholds, so recorded as such.
CLEAN_UTIL_MARGIN = 10.0
CLEAN_MEM_MARGIN_GIB = 1.0

_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu"


def parse_nvidia_smi(text: str) -> dict | None:
    """The first parseable `nvidia-smi --query-gpu=<_QUERY> --format=csv,noheader,nounits` line, or None.

    A field nvidia-smi cannot read prints as `[N/A]` and makes its line unparseable, so the line is skipped rather
    than read as zero."""
    for line in text.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            util, used, total, temp = (float(x) for x in parts)
        except ValueError:
            continue
        return {"util": util, "mem_used_mib": used, "mem_total_mib": total, "temp_c": temp}
    return None


def gpu_reading() -> dict | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_nvidia_smi(out)


def summarise_samples(samples: list[dict]) -> dict:
    """Mean and peak utilisation and memory, peak temperature, over a list of readings."""
    if not samples:
        return {"available": False, "n": 0}
    return {
        "available": True,
        "n": len(samples),
        "util_mean": statistics.fmean(s["util"] for s in samples),
        "util_max": max(s["util"] for s in samples),
        "mem_used_mib_mean": statistics.fmean(s["mem_used_mib"] for s in samples),
        "mem_used_mib_max": max(s["mem_used_mib"] for s in samples),
        "mem_total_mib": samples[0]["mem_total_mib"],
        "temp_c_max": max(s["temp_c"] for s in samples),
    }


def background_load(seconds: float = 10.0, every: float = 0.5) -> dict:
    """Sample the GPU for `seconds` while the caller runs nothing of ours. Ten seconds by default, because the
    foreign job is bursty and a short window can land entirely inside or outside a burst."""
    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        r = gpu_reading()
        if r is not None:
            samples.append(r)
        time.sleep(every)
    out = summarise_samples(samples)
    out["seconds"] = seconds
    out["at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return out


def is_clean(bg: dict, reference: dict) -> tuple[bool, str]:
    """(clean, reason) for ONE background reading against the reference. Unavailable readings are not clean,
    because cleanliness cannot then be shown. Both margins are inclusive: a difference EQUAL to a margin is clean."""
    if not bg.get("available") or not reference.get("available"):
        return False, "no GPU reading, so the background load is unknown"
    du = abs(bg["util_mean"] - reference["util_mean"])
    dm = abs(bg["mem_used_mib_mean"] - reference["mem_used_mib_mean"]) / 1024.0
    reasons = []
    if du > CLEAN_UTIL_MARGIN:
        reasons.append(
            f"background utilisation {bg['util_mean']:.0f} percent against {reference['util_mean']:.0f}"
        )
    if dm > CLEAN_MEM_MARGIN_GIB:
        reasons.append(f"background memory moved by {dm:.2f} GiB")
    return (not reasons), ("; ".join(reasons) or "background load matches the reference")


def round_clean(before: dict, after: dict, reference: dict) -> tuple[bool, str]:
    """A round is clean only if the background BEFORE it and AFTER it both match the reference. The foreign job
    can start or stop while a round runs; checking both ends catches that, which checking one end cannot."""
    ok_b, why_b = is_clean(before, reference)
    ok_a, why_a = is_clean(after, reference)
    if ok_b and ok_a:
        return True, "background before and after matches the reference"
    parts = []
    if not ok_b:
        parts.append(f"before: {why_b}")
    if not ok_a:
        parts.append(f"after: {why_a}")
    return False, "; ".join(parts)


class Sampler:
    """GPU and host readings every `every` seconds on a thread, while a round runs."""

    def __init__(self, every: float = 1.0):
        self.every, self.gpu, self.load, self.mem_avail = every, [], [], []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            r = gpu_reading()
            if r is not None:
                self.gpu.append(r)
            h = host_snapshot()
            self.load.append(h["load1"])
            self.mem_avail.append(h["mem_available_gib"])
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=15)

    def result(self) -> dict:
        out = summarise_samples(self.gpu)
        out["host_load1_max"] = max(self.load) if self.load else None
        out["host_mem_available_gib_min"] = min(self.mem_avail) if self.mem_avail else None
        return out


def host_snapshot() -> dict:
    load1 = float("nan")
    try:
        load1 = float(pathlib.Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    mem = float("nan")
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem = float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return {"nproc": os.cpu_count(), "load1": load1, "mem_available_gib": mem}


# ---------------------------------------------------------------------------------- the compilation cache ----
#: JAX 0.11.0's LRU cache writes `<key>-cache` and `<key>-atime` (read from `jax._src.lru_cache`, not assumed),
#: holds `.lockfile` while it reads or writes, and XLA's per-fusion autotune cache lives in this subdirectory.
CACHE_SUFFIX, ATIME_SUFFIX, LOCKFILE, AUTOTUNE_DIR = (
    "-cache",
    "-atime",
    ".lockfile",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


def cache_census(
    cache_dir: pathlib.Path, since: datetime.datetime | None = None, max_size: int | None = None
) -> dict:
    """What the persistent compilation cache holds, and the two states that silently stop it writing.

    **Orphans.** `LRUCache._evict_if_needed` runs on every write when a maximum size is set (this project sets 2 GB)
    and reads the `-atime` file of EVERY `-cache` entry. A `-cache` entry without its `-atime` file makes that read
    raise, the write fails, and with `jax_raise_persistent_cache_errors` False (its default) the error is swallowed:
    from then on the cache is read but never written. So `-cache` files lacking an `-atime` are counted and named.
    An `-atime` without its `-cache` is harmless and only counted.

    **Size.** A cache at or above `max_size` evicts on every write; an entry larger than `max_size` is refused."""
    d = pathlib.Path(cache_dir)
    if not d.is_dir():
        return {"dir": str(d), "exists": False}
    cache, atime, other = {}, {}, []
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.name.endswith(CACHE_SUFFIX):
            cache[p.name[: -len(CACHE_SUFFIX)]] = p
        elif p.name.endswith(ATIME_SUFFIX):
            atime[p.name[: -len(ATIME_SUFFIX)]] = p
        else:
            other.append(p.name)
    iso = lambda t: datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")  # noqa: E731
    cmt = [p.stat().st_mtime for p in cache.values()]
    amt = [p.stat().st_mtime for p in atime.values()]
    sub = d / AUTOTUNE_DIR
    sub_files = [p for p in sub.rglob("*") if p.is_file()] if sub.is_dir() else []
    orphans = sorted(k for k in cache if k not in atime)
    size = sum(p.stat().st_size for p in cache.values())
    out = {
        "dir": str(d),
        "exists": True,
        "cache_entries": len(cache),
        "atime_files": len(atime),
        "cache_without_atime": len(orphans),
        "cache_without_atime_names": orphans[:20],
        "atime_without_cache": sum(k not in cache for k in atime),
        "lockfile_present": LOCKFILE in other,
        "other_files": sorted(other),
        "cache_bytes": size,
        "newest_entry": iso(max(cmt)) if cmt else None,
        "newest_read": iso(max(amt)) if amt else None,
        "autotune_files": len(sub_files),
        "autotune_newest": iso(max(p.stat().st_mtime for p in sub_files)) if sub_files else None,
        "disk_free_gib": shutil.disk_usage(d).free / 2**30,
    }
    if max_size is not None:
        out["max_size"] = int(max_size)
        out["fraction_of_max"] = size / max_size if max_size > 0 else None
    if since is not None:
        ts = since.timestamp()
        out["since"] = since.isoformat(timespec="seconds")
        out["entries_written_since"] = sum(t >= ts for t in cmt)
        out["entries_read_since"] = sum(t >= ts for t in amt)
    return out
