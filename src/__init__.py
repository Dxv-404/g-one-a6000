"""Runtime environment control. Importing anything from this package triggers it.

**`uv sync` reproduces packages. It does not reproduce environment variables, and several of JAX's change the
numerics.** Setting them in `tests/conftest.py` alone covers pytest and nothing else: a throughput run, an
evolution run, an interactive session, or `python configs/default.py` launched directly would all bypass it.
Since the preallocation flag has no effect once JAX has initialised, a measurement run launched outside pytest
would silently use a different memory ceiling than the one every gate was verified against, and M2 would then
be measured against a ceiling no reported result actually ran under.

So the setting lives here, at package import, and `tests/conftest.py` imports this rather than duplicating it.
One implementation, triggered by any entry point that touches the project.

    XLA_PYTHON_CLIENT_PREALLOCATE   sets the memory ceiling. Hard-set, not defaulted
    XLA_PYTHON_CLIENT_ALLOCATOR     cuda_async. Hard-set. JAX 0.11's default BFC allocator hangs forever on
                                    its first allocation on the A6000 under WSL; see D-067
    XLA_PYTHON_CLIENT_MEM_FRACTION  same, continuously. Left alone; D-008 settled that it stays default
    JAX_ENABLE_X64                  switches float32 to float64, invalidating every tolerance. Refused
    JAX_PLATFORMS                   can force CPU, which looks like a slow GPU rather than a wrong one

**Setting a variable early is a convention. Checking that it took effect is a guarantee.**
`assert_runtime_took_effect()` verifies the observable consequences after JAX is up rather than trusting that
the assignment happened first. Call it from anything that produces a number: the gates call it through
conftest, and every measurement entry point from step 11 onward must call it too.

**The dangerous ordering is narrower than "import jax first", and the difference is worth being exact about.**
JAX reads these flags when the backend initialises, which happens at the first *operation*, not at import. So
`import jax` before `import src` is harmless as long as nothing has run. What defeats the setting is a JAX
operation preceding the package import. That is a smaller and more checkable surface than import order in
general, and it is the case the runtime assertion exists for. Established by measurement rather than by
reading; see D-022.

This module must not import jax. Importing it here would not by itself break anything, since import alone does
not initialise the backend, but it would make package import slow and would put an operation one careless line
away from running before the flags are set.
"""

import os

# Hard-set rather than defaulted. Deferring to a shell value would mean a clone and this machine disagree
# about what "default" means, which is the failure being guarded against rather than a courtesy.
REQUIRED = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    # D-067. The default BFC allocator in JAX 0.11 maps device memory through CUDA's virtual-memory API and
    # asks for RDMA-capable memory whenever the driver reports support. The A6000 reports it; under WSL the
    # mapping then fails in cuMemSetAccess with CUDA_ERROR_UNKNOWN, BFC reads that as out of memory, and it
    # retries ever smaller sizes forever without raising. cuda_async allocates from CUDA's stream-ordered pool
    # and never takes that path. Allocation does not touch arithmetic, so no result changes; E1 confirms it.
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
}

FORBIDDEN = {
    "JAX_ENABLE_X64": "would switch float32 to float64 and invalidate every float tolerance",
}

# What the shell supplied, before anything here touched it. Kept so an override is visible rather than
# silent, and so task 6's environment capture can record what was actually in effect rather than what this
# module intended.
INHERITED_ENVIRONMENT = {
    k: os.environ.get(k)
    for k in (
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_CLIENT_MEM_FRACTION",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
    )
}

# --------------------------------------------------------------------------- compilation cache
#
# XLA compilation, not execution, is almost all of this project's test runtime: 147 tests in 890 seconds is
# 6.1 seconds each, on grid operations over arrays whose actual compute is microseconds. Measured, two runs of
# the same workload in separate processes:
#
#     without a persistent cache    first 7.7s   second 5.1s
#     with one                      first 4.1s   second 2.0s
#
# The cache was not configured at all, so every run recompiled everything from scratch. That matters beyond
# convenience: a suite slow enough to be run reluctantly stops being run, and an engineering-only habit has
# already hidden a red test for a full session in this project.
#
# **The directory must not sit under /mnt/c.** Measured on this machine, small-file creation there runs at 155
# files per second against 55,915 on the WSL filesystem, and a compile cache is exactly that access pattern,
# so a cache on the Windows mount would be slower than no cache. Asserted below rather than assumed.
#
# Kept outside the repository, in the conventional user cache location, so there is nothing to gitignore and
# nothing to exclude from the wheel. It is named for the project so clones do not collide with other JAX work.
_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "jax-emergent-self-preservation",
)

if os.environ.get("JAX_COMPILATION_CACHE_DIR") is None:
    if _CACHE_DIR.startswith("/mnt/"):
        raise RuntimeError(
            f"the compilation cache would be at {_CACHE_DIR}, on a Windows mount, where small-file creation "
            "is roughly 360 times slower than on the WSL filesystem. A cache there is worse than none. Set "
            "XDG_CACHE_HOME to a path under /home, or set JAX_COMPILATION_CACHE_DIR explicitly."
        )
    # Fail loudly rather than silently. JAX degrades quietly when a cache directory is unusable, and a
    # silently disabled cache looks exactly like no cache, which is the same failure class as an inert
    # assertion: no error, no signal, and a timing figure that means something different from what it says.
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        _probe = os.path.join(_CACHE_DIR, ".writable")
        with open(_probe, "w") as _f:
            _f.write("x")
        os.unlink(_probe)
    except OSError as _exc:
        raise RuntimeError(
            f"the compilation cache directory {_CACHE_DIR} is not writable: {_exc}. JAX would disable the "
            "cache silently, so every run would recompile from scratch while appearing configured, and any "
            "timing recorded against it would be wrong. Fix the path or set JAX_COMPILATION_CACHE_DIR."
        ) from _exc

    os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR
    # Defaults skip small or fast-to-compile executables, which is most of this project's kernels. Zero means
    # cache everything, which is what makes repeat runs fast here rather than marginally faster.
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    # JAX does not evict by default: `jax_compilation_cache_max_size` is -1, meaning unbounded, so the
    # directory would grow for the life of the project. Measured, the growth is small: one full suite writes
    # 1619 entries totalling 7.1 MB, a mean of 4.4 kB each, against 1 TB free.
    #
    # So this is insurance rather than a fix for an observed problem. It is worth one line because the M4
    # sweep will compile a new set of programs for every distinct shape it visits, and the failure mode of
    # an unbounded cache is a full disk at an inconvenient moment rather than anything subtle. 2 GB is about
    # 285 times the current size, so it will not evict anything in normal use.
    #
    # The cap requires `filelock`, which is present as a transitive dependency. If it ever disappears the
    # cap is silently ignored, which is why the size is checked in task 6's environment capture rather than
    # assumed to be holding.
    os.environ.setdefault("JAX_COMPILATION_CACHE_MAX_SIZE", str(2 * 1024**3))

_overridden = []
for _name, _value in REQUIRED.items():
    _previous = os.environ.get(_name)
    if _previous is not None and _previous != _value:
        _overridden.append(f"{_name}={_previous!r} overridden to {_value!r}")
    os.environ[_name] = _value

for _name, _why in FORBIDDEN.items():
    _present = os.environ.get(_name)
    if _present not in (None, "", "0", "false", "False"):
        raise RuntimeError(
            f"{_name} is set to {_present!r} in the environment, which {_why}. "
            "Unset it and re-run. The gates were written and verified without it, so a run with it set is "
            "not comparable to any recorded result."
        )

OVERRIDDEN = tuple(_overridden)


def memory_headroom(min_available_gb: float = 3.0, *, raise_on_low: bool = False) -> dict:
    """Report WSL's available memory before a long run, and optionally refuse to start.

    **Why this exists and what it does NOT do.** DEV-029 was a leak inside the generation loop; the
    freeze-pricing scripts met the same class outside it and mitigated with `jax.clear_caches()` per
    seed. Both are about the GUEST's memory. **Neither looks at whether the run fits alongside
    everything else on the machine**, and on 2026-08-31 the host bugchecked with 0x1A
    MEMORY_MANAGEMENT two minutes after the differential tier finished, with 0.11 GB of 15.71 GB free.
    See D-054.

    **This reads the guest, because the guest is what it can read.** WSL cannot see Windows' free
    memory, so this is a proxy: if WSL itself is near its ceiling it is certainly holding memory the
    host cannot use, since WSL2 does not return freed pages promptly without `autoMemoryReclaim`. A
    comfortable reading here does NOT prove the host has headroom, and saying so is the point of the
    docstring rather than a caveat buried in a comment.

    Defaults to reporting rather than raising, because a measurement script that refuses to start on a
    busy machine is worse than one that says what it saw.
    """
    info = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                info[k.strip()] = float(v.strip().split()[0]) / (1024.0 * 1024.0)  # kB -> GiB
    except OSError:
        return {"available_gb": float("nan"), "total_gb": float("nan"), "ok": True}

    out = {
        "available_gb": info.get("MemAvailable", float("nan")),
        "total_gb": info.get("MemTotal", float("nan")),
        "used_gb": info.get("MemTotal", 0.0) - info.get("MemAvailable", 0.0),
    }
    out["ok"] = not (out["available_gb"] < min_available_gb)
    if not out["ok"]:
        msg = (
            f"host headroom low: {out['available_gb']:.1f} GiB available of {out['total_gb']:.1f} inside "
            f"WSL, against a floor of {min_available_gb:.1f}. WSL does not return freed pages to Windows "
            "promptly, so this is also the host's problem. See D-054."
        )
        if raise_on_low:
            raise SystemExit(msg)
        print(f"WARNING: {msg}", flush=True)
    return out


def assert_runtime_took_effect() -> dict:
    """Verify the runtime JAX actually has, not the one this module asked for. Returns what was observed.

    Checks consequences rather than environment variables, because the failure mode is an import ordering in
    which the variables were set too late. A late assignment leaves the variable looking correct in
    `os.environ` while JAX has already initialised without it.

    Two observable properties:

    **The allocator is cuda_async, and preallocation is off.** Both read from `memory_stats()` by
    `check_memory_stats`, which the gates call too, so a test cannot pass against a copy of the logic. The
    metrics trade roles between allocators, which is the trap worth naming:

        under BFC (before D-067)
            bytes_limit   75 percent of the card whether preallocation is on or off. D-008's finding
            pool_bytes    the discriminator. Measured 2 MB off against 4226 MB on
        under cuda_async (measured on the A6000, D-067)
            pool_bytes    absent. Its presence means BFC is running, so the allocator setting did not take
            bytes_limit   the discriminator. Measured 0 off against 34550 MiB on, 75 percent of 46067
            bytes_in_use  live allocations; zero after a trivial allocation under both settings, so inert

    `nvidia-smi` per-process memory would also show it in principle and reports `[N/A]` under WSL, so it is
    not an option here.

    **float32 is the default.** Asserted on an actual array dtype rather than on `JAX_ENABLE_X64`, so it also
    catches x64 enabled programmatically through `jax.config.update`, which no environment check would see.
    """
    import jax
    import jax.numpy as jnp

    observed = {"platform": jax.devices()[0].platform}

    probe = jnp.zeros(1024, dtype=jnp.float32)
    probe.block_until_ready()
    observed["default_float_dtype"] = str(jnp.zeros(1).dtype)

    if observed["default_float_dtype"] != "float32":
        raise RuntimeError(
            f"JAX default float dtype is {observed['default_float_dtype']}, not float32. x64 is enabled "
            "somewhere, and every differential tolerance and pool comparison was written against float32. "
            "No result from this process is comparable to a recorded one."
        )

    stats = jax.devices()[0].memory_stats() if observed["platform"] == "gpu" else None
    if stats:
        observed.update(check_memory_stats(stats))
    return observed


def check_memory_stats(stats: dict) -> dict:
    """Raise unless `stats` comes from the cuda_async allocator with preallocation off. Returns what it read.

    Pure, so the gates can hand it a BFC-shaped or preallocated dict and see it refuse, and so the guard and
    the gates run one implementation rather than two that could drift. Measured values are in the docstring
    of `assert_runtime_took_effect`.
    """
    if "pool_bytes" in stats:
        raise RuntimeError(
            f"memory_stats() reports pool_bytes={stats['pool_bytes']}, which only the BFC allocator does. "
            "XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async did not take effect, so this process allocates through the "
            "path that hangs on the A6000 under WSL (D-067) and differs from the one the gates were verified "
            "under. The likely cause is a JAX operation running before `src` was imported."
        )
    limit = int(stats.get("bytes_limit", -1))
    if limit != 0:
        raise RuntimeError(
            f"the cuda_async allocator reports bytes_limit={limit}, against 0 when preallocation is off "
            "(measured 34550 MiB with it on, D-067). Preallocation is on, so this process has a different memory "
            "ceiling than the one the gates were verified against and than the one M2 records. The likely cause "
            "is a JAX operation running before `src` was imported: the flag is read when the backend "
            "initialises, which is the first operation rather than the import."
        )
    return {"allocator": "cuda_async", "bytes_limit": limit}
