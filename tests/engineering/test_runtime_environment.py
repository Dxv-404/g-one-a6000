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
    }
    assert set(src.FORBIDDEN) == {"JAX_ENABLE_X64"}


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
