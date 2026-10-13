"""Runtime environment control for test runs. Imported by pytest before collection.

The logic lives in `src/__init__.py` rather than here, and this file imports it. Duplicating it would give two
implementations that drift, and more importantly a conftest-only version covers pytest and nothing else: a
throughput run, an evolution run, or `python configs/default.py` launched directly would all bypass it and get
a different memory ceiling. Since the preallocation flag has no effect once JAX has initialised, that
difference would be silent.

Importing `src` here is what triggers the setting, and it happens before any test module is imported.
"""

import src  # noqa: F401  imported for its import-time side effect, which is the entire point


def pytest_report_header(config):
    """Surface the runtime environment at the top of every run, and verify it took effect.

    Printed rather than merely checked, because the failure being guarded against is one where everything
    passes and the numbers are different. The header is where someone would notice.

    The assertion runs here rather than in a fixture so that a misconfigured runtime stops the session before
    any gate reports green against it.
    """
    lines = []

    inherited = {k: v for k, v in src.INHERITED_ENVIRONMENT.items() if v is not None}
    lines.append(
        f"runtime: inherited from shell: {inherited}"
        if inherited
        else "runtime: nothing inherited from the shell, which is the intended state"
    )
    if src.OVERRIDDEN:
        lines.append(f"runtime OVERRODE: {'; '.join(src.OVERRIDDEN)}")

    try:
        observed = src.assert_runtime_took_effect()
    except Exception as exc:  # surfaced as a header line and re-raised, so it cannot be missed
        lines.append(f"runtime CHECK FAILED: {exc}")
        raise

    if "bytes_limit" in observed:
        lines.append(
            f"runtime verified: {observed['platform']}, {observed['default_float_dtype']}, "
            f"allocator pool {observed['pool_bytes'] / 1e6:.0f} MB of a "
            f"{observed['bytes_limit'] / 1e6:.0f} MB ceiling"
        )
    else:
        lines.append(f"runtime verified: {observed['platform']}, {observed['default_float_dtype']}")
    return lines
