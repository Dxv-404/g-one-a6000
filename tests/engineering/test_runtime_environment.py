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
    assert src.REQUIRED == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
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
    """Checked against `pool_bytes`, which is the only one of the three plausible metrics that can see it.

    Two wrong answers, both of which look right, and both were tried:

        bytes_limit   75 percent of the card whether preallocation is on or off. D-008's measured finding
        bytes_in_use  live allocations only. Zero under both settings, so a check against it passes on a
                      misconfigured run and is inert by construction rather than by ordering
        pool_bytes    reserved by the allocator. Measured at 2 MB off against 4226 MB on

    The first version of this test used `bytes_in_use` and was inert. It was caught by trying to break it and
    finding it would not go red, which is the whole reason a check gets broken deliberately rather than
    reasoned about.
    """
    import jax

    if jax.devices()[0].platform != "gpu":
        return
    stats = jax.devices()[0].memory_stats()
    pool, limit = stats["pool_bytes"], stats["bytes_limit"]
    assert pool < 0.5 * limit, (
        f"allocator pool is {pool / 1e6:.0f} MB of a {limit / 1e6:.0f} MB ceiling. Preallocation is on, so "
        "this process has a different memory ceiling than every recorded result."
    )


def test_the_check_is_reachable_from_outside_pytest():
    """Every measurement entry point from step 11 must call this, so it has to be importable from `src`.

    Asserted because the whole point of moving the logic out of conftest was that a conftest-only version
    covers pytest and nothing else.
    """
    assert callable(src.assert_runtime_took_effect)
    assert hasattr(src, "INHERITED_ENVIRONMENT")
    assert hasattr(src, "OVERRIDDEN")
