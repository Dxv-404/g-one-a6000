"""The runtime guarantee itself: that the environment JAX has is the one the project requires.

Separate from gate E0, which checks the machine is capable. This checks the process is configured, which is a
different failure: E0 would pass on a run whose memory ceiling differs from every recorded result, because the
GPU is present and fast either way.

The distinction that makes this worth having: setting an environment variable early is a convention that holds
until someone imports in a different order, and `import jax` before `import src` would defeat it silently.
Checking the observable consequence is a guarantee.
"""

import jax.numpy as jnp

import src


def test_the_forbidden_and_required_sets_are_what_the_project_requires():
    """Transcribed rather than imported, so editing one without the other fails here."""
    assert src.REQUIRED == {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
        "XLA_FLAGS": "--xla_gpu_autotune_level=0",
    }
    assert set(src.FORBIDDEN) == {"JAX_ENABLE_X64"}


def test_the_compiled_program_is_pinned_in_this_process():
    """DEV-045: what XLA reads at backend start, not what the dict says. JAX reads XLA_FLAGS when the backend
    initialises, which `import src` precedes, so the process environment at that point is the evidence."""
    import os

    assert os.environ.get("XLA_FLAGS") == "--xla_gpu_autotune_level=0"


def test_runtime_check_passes_and_reports_what_it_observed():
    observed = src.assert_runtime_took_effect()
    assert observed["default_float_dtype"] == "float32"
    assert observed["platform"] in ("gpu", "cpu")


def test_default_float_is_float32_not_float64():
    """Asserted on an array rather than on the environment variable.

    x64 can be enabled programmatically through `jax.config.update`, which no environment check would see, and
    it would change every differential tolerance and pool comparison underneath tests written against float32.
    """
    assert jnp.zeros(1).dtype == jnp.float32
    assert jnp.array(1.0).dtype == jnp.float32


def test_preallocation_is_off_observably():
    """Checked through `src.check_memory_stats`, the guard's own function, on this process's real stats.

    Under cuda_async (D-067) the discriminator is `bytes_limit`: measured 0 off against 34550 MiB on. Under
    BFC it was `pool_bytes`, and `bytes_limit` was the inert one, so the metric that was wrong before is the
    one that is right now. The history of wrong answers, kept because each looked right:

        bytes_in_use  live allocations only. Zero under both settings. The first version of this test used it
                      and was inert, caught by trying to break it and finding it would not go red
        bytes_limit   inert under BFC (D-008), the discriminator under cuda_async
        pool_bytes    the discriminator under BFC, absent under cuda_async
    """
    import jax

    if jax.devices()[0].platform != "gpu":
        return
    observed = src.check_memory_stats(jax.devices()[0].memory_stats())
    assert observed == {"allocator": "cuda_async", "bytes_limit": 0}


def test_the_allocator_check_refuses_bfc_and_preallocation():
    """The check must be able to fail, and it cannot be broken for real on the A6000: BFC hangs there.

    So it is handed the two dicts it exists to refuse, shaped as measured: BFC's, with `pool_bytes` and its
    constant 75 percent `bytes_limit` from the laptop, and cuda_async's with preallocation on. Same function
    the guard runs, so this is coupled to the mechanism rather than to a copy of it.
    """
    import pytest

    bfc = {"bytes_in_use": 4096, "pool_bytes": 2 * 2**20, "bytes_limit": 6 * 2**30}
    with pytest.raises(RuntimeError, match="only the BFC allocator"):
        src.check_memory_stats(bfc)

    preallocated = {"bytes_in_use": 0, "peak_bytes_in_use": 0, "bytes_limit": 34550 * 2**20}
    with pytest.raises(RuntimeError, match="Preallocation is on"):
        src.check_memory_stats(preallocated)

    healthy = {"bytes_in_use": 0, "peak_bytes_in_use": 0, "bytes_limit": 0, "bytes_reserved": 0}
    assert src.check_memory_stats(healthy)["allocator"] == "cuda_async"


def test_the_check_is_reachable_from_outside_pytest():
    """Every measurement entry point from step 11 must call this, so it has to be importable from `src`.

    Asserted because the whole point of moving the logic out of conftest was that a conftest-only version
    covers pytest and nothing else.
    """
    assert callable(src.assert_runtime_took_effect)
    assert hasattr(src, "INHERITED_ENVIRONMENT")
    assert hasattr(src, "OVERRIDDEN")


# ------------------------------------------------------------------------------------- the cache, DEV-039 ----
_JAX_FIRST_PROBE = """
import os, jax
import src
f = jax.jit(lambda x: x * 3.0 + 1.0)
f(jax.numpy.ones(5)).block_until_ready()
d = jax.config.jax_compilation_cache_dir
print("DIR", d)
print("FILES", sum(len(fs) for _, _, fs in os.walk(d)) if d else -1)
"""


def test_the_cache_works_when_jax_is_imported_before_src(tmp_path):
    """DEV-039. Every measurement driver imported `jax` before `src`, JAX read the cache settings at import, and
    the drivers compiled without a cache for two months while every check passed. Asserted on BEHAVIOUR: a
    program compiled in a process that imports `jax` first must land in the cache `src` configures.

    The cache home is a fresh directory so the count can only come from this process, and
    JAX_COMPILATION_CACHE_DIR is removed so `src` sets it, which is the ordering the defect needed."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_")}
    env["XDG_CACHE_HOME"] = str(tmp_path)
    out = subprocess.run(
        [sys.executable, "-c", _JAX_FIRST_PROBE], env=env, capture_output=True, text=True, check=True
    )
    lines = dict(line.split(" ", 1) for line in out.stdout.splitlines() if line.startswith(("DIR", "FILES")))
    assert lines["DIR"].startswith(str(tmp_path)), (
        lines
    )  # the probe saw the cache src configured, not another
    assert int(lines["FILES"]) > 0, f"nothing written to the cache with jax imported first: {lines}"


def test_the_runtime_check_refuses_a_process_without_the_cache():
    """The pure check, handed what a jax-first process had before the fix: no directory, default thresholds."""
    import pytest

    want = {
        "JAX_COMPILATION_CACHE_DIR": "/home/x/.cache/jax-emergent-self-preservation",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    }
    stale = {"jax_compilation_cache_dir": None, "jax_persistent_cache_min_compile_time_secs": 1.0}
    with pytest.raises(RuntimeError, match="compilation cache"):
        src.check_cache_config(stale, want)
    wrong_threshold = {
        "jax_compilation_cache_dir": want["JAX_COMPILATION_CACHE_DIR"],
        "jax_persistent_cache_min_compile_time_secs": 1.0,
    }
    with pytest.raises(RuntimeError, match="min_compile_time"):
        src.check_cache_config(wrong_threshold, want)
    with pytest.raises(RuntimeError, match="no compilation cache directory"):
        src.check_cache_config({"jax_compilation_cache_dir": None}, {})
    good = {
        "jax_compilation_cache_dir": want["JAX_COMPILATION_CACHE_DIR"],
        "jax_persistent_cache_min_compile_time_secs": 0.0,
    }
    assert src.check_cache_config(good, want)["cache_dir"] == want["JAX_COMPILATION_CACHE_DIR"]


def test_this_process_has_the_cache_configured():
    assert src.assert_runtime_took_effect()["cache_dir"]


# ------------------------------------------------------------------- the startup race, DEV-040 ----
def test_two_writability_probes_never_share_a_file(tmp_path):
    """DEV-040. The probe was a fixed `.writable`, so two processes starting together could each create it and
    the second to delete it died. Two probes must use different files, and neither may be left behind."""
    a = src.probe_writable(str(tmp_path))
    b = src.probe_writable(str(tmp_path))
    assert a != b
    assert not list(tmp_path.iterdir()), "a probe file was left behind"


_RACER = """
import os, sys, time
go = sys.argv[1]
while not os.path.exists(go):
    time.sleep(0.001)
import src
print("OK")
"""


def test_many_processes_starting_at_once_all_get_past_the_runtime_control(tmp_path):
    """The behavioural witness: 16 processes released at the same instant against one fresh cache home, the
    situation the A6000's six-process round met. Every one must start. Probabilistic on its own, which is why
    the deterministic test above exists; this one checks the cure in the setting where the defect appeared."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_")}
    env["XDG_CACHE_HOME"] = str(tmp_path / "home")
    go = tmp_path / "GO"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _RACER, str(go)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(16)
    ]
    go.write_text("go")
    results = [p.communicate(timeout=120) for p in procs]
    failed = [
        err[-300:]
        for (out, err), p in zip(results, procs, strict=True)
        if p.returncode != 0 or "OK" not in out
    ]
    assert not failed, f"{len(failed)} of 16 processes died at startup: {failed[:1]}"
