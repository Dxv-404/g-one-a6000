"""Capture the environment a measurement was produced in, automatically.

The publication-readiness list requires compute hours, exact software versions, and the environment version
recorded with every reported result, and **none of it can be reconstructed after the fact.** A result from a
dirty working tree is not reproducible and must be recorded as such at the time rather than discovered later.

This writes into the header of every measurement output from step 11 onward.

### Two fields that are easy to get wrong

**Cold versus warm compilation cache is not observable after the fact**, so the entry count is snapshotted at
module import, before any measurement has had a chance to add to it. Inferring warmth from how long a run
took would be worse than not recording it: a run slow for an unrelated reason would be labelled cold
permanently in the record. Entry count at start is a fact; runtime is an inference.

**The environment values recorded are the ones actually in effect, not the ones the code intended to set.**
Those differ precisely when something external has overridden them, which is the case worth catching.
`src.INHERITED_ENVIRONMENT` holds what the shell supplied and `src.OVERRIDDEN` holds what was overridden.

### The failure mode this must not have

A capture that returns a plausible-looking dictionary of defaults when its underlying calls fail: every field
populated, nothing true. That is the construction class, and it would be invisible, because a header reading
`driver: unknown, dirty: False` looks exactly like a clean environment. So every subprocess call records its
failure explicitly rather than falling back to a neutral-looking value, and `capture()` sets `complete` to
False if anything could not be determined.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

import src

REPO = Path(__file__).resolve().parents[2]

# Snapshotted at import, before any measurement runs. See the module docstring.
_CACHE_DIR = os.environ.get("JAX_COMPILATION_CACHE_DIR")
_CACHE_AT_START = None
if _CACHE_DIR and os.path.isdir(_CACHE_DIR):
    _entries = [p for p in Path(_CACHE_DIR).glob("*") if p.is_file()]
    _CACHE_AT_START = {
        "entries": len(_entries),
        "bytes": sum(p.stat().st_size for p in _entries),
    }

# A sentinel distinguishable from a real value. Never substitute a plausible default: "unknown" in a header is
# a question, whereas "570.0" that nobody measured is an answer, and a wrong one.
UNKNOWN = "UNDETERMINED"


def _run(cmd, cwd=None):
    """Run a command, returning its stdout or an explicit failure marker. Never a plausible default."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{UNKNOWN}: {type(exc).__name__}"
    if out.returncode != 0:
        return f"{UNKNOWN}: exit {out.returncode}"
    return out.stdout.strip()


def _packages():
    """Every installed distribution and version.

    `importlib.metadata`, not `pip list`: uv environments do not install pip, so `pip list` returns nothing
    and the capture would record an empty environment as though that were a fact.
    """
    import importlib.metadata as md

    return {d.metadata["Name"].lower(): d.version for d in md.distributions() if d.metadata["Name"]}


def _gpu():
    """GPU model, driver, and total memory, from nvidia-smi."""
    raw = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if raw.startswith(UNKNOWN):
        return {"model": raw, "driver": raw, "memory_total": raw}
    name, driver, total = (f.strip() for f in raw.split(",")[:3])
    return {"model": name, "driver": driver, "memory_total": total}


def _git():
    """Commit hash and whether the working tree was dirty.

    **A dirty tree is recorded, never silently tolerated.** A run from a dirty tree is not reproducible from
    its commit, and `runs.md` requires it marked as such rather than discovered eighteen months later.
    """
    commit = _run(["git", "rev-parse", "HEAD"], cwd=REPO)
    status = _run(["git", "status", "--porcelain"], cwd=REPO)
    if status.startswith(UNKNOWN):
        return {"commit": commit, "dirty": UNKNOWN, "dirty_files": UNKNOWN}
    files = [line for line in status.splitlines() if line.strip()]
    return {"commit": commit, "dirty": bool(files), "dirty_files": len(files)}


def _jax_runtime():
    """What JAX actually has, verified rather than assumed. Raises if the runtime is misconfigured."""
    observed = src.assert_runtime_took_effect()
    import jax

    return {
        "jax_version": jax.__version__,
        "platform": observed["platform"],
        "default_float_dtype": observed["default_float_dtype"],
        "pool_bytes": observed.get("pool_bytes", UNKNOWN),
        "bytes_limit": observed.get("bytes_limit", UNKNOWN),
    }


def capture() -> dict:
    """The environment this process is running in. Call from every measurement entry point.

    `complete` is False if any field could not be determined, so a consumer can tell a captured environment
    from a partially captured one without inspecting every value.
    """
    packages = _packages()
    gpu = _gpu()
    git = _git()

    cache = {
        "dir": _CACHE_DIR or UNKNOWN,
        "state_at_start": "warm" if (_CACHE_AT_START and _CACHE_AT_START["entries"]) else "cold",
        "entries_at_start": _CACHE_AT_START["entries"] if _CACHE_AT_START else UNKNOWN,
        "bytes_at_start": _CACHE_AT_START["bytes"] if _CACHE_AT_START else UNKNOWN,
        # The 2 GB cap needs filelock and is silently ignored without it, so record whether it can apply.
        "max_size_enforceable": "filelock" in packages,
        "max_size": os.environ.get("JAX_COMPILATION_CACHE_MAX_SIZE", UNKNOWN),
    }

    env = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "os": platform.platform(),
        "packages": packages,
        "n_packages": len(packages),
        "gpu": gpu,
        "git": git,
        "cache": cache,
        "jax": _jax_runtime(),
        # What the shell supplied and what was overridden, rather than what the code meant to set.
        "inherited_environment": {k: v for k, v in src.INHERITED_ENVIRONMENT.items() if v is not None},
        "overridden": list(src.OVERRIDDEN),
    }

    def _has_unknown(value):
        if isinstance(value, str):
            return value.startswith(UNKNOWN)
        if isinstance(value, dict):
            return any(_has_unknown(v) for v in value.values())
        if isinstance(value, list):
            return any(_has_unknown(v) for v in value)
        return False

    env["complete"] = not _has_unknown(env)
    return env


def header_lines(env=None) -> list[str]:
    """A compact human-readable form, for the top of a measurement output file."""
    e = env if env is not None else capture()
    g, gt, c = e["gpu"], e["git"], e["cache"]
    lines = [
        f"python {e['python']}   jax {e['jax']['jax_version']}   {e['jax']['platform']}, "
        f"{e['jax']['default_float_dtype']}",
        f"gpu {g['model']}   driver {g['driver']}   memory {g['memory_total']}",
        f"commit {gt['commit']}   working tree {'DIRTY' if gt['dirty'] is True else 'clean'}"
        + (f" ({gt['dirty_files']} files)" if gt["dirty"] is True else ""),
        f"compilation cache {c['state_at_start']} at start, {c['entries_at_start']} entries",
        f"packages {e['n_packages']}   capture {'complete' if e['complete'] else 'INCOMPLETE'}",
    ]
    if e["inherited_environment"]:
        lines.append(f"inherited from shell: {e['inherited_environment']}")
    if gt["dirty"] is True:
        lines.append("WARNING: a result from a dirty working tree is not reproducible from its commit")
    return lines
