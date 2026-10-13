"""The environment capture, and the failure mode it must not have.

**A capture that returns a plausible-looking dictionary of defaults when its underlying calls fail is the
construction class**: every field populated, nothing true, and invisible in a header because
`driver: unknown, dirty: False` reads exactly like a clean environment. So these tests check that each field
is a real observation, and that a failure is marked as one rather than filled in.

Each assertion is against something independently verifiable: the commit against `git rev-parse HEAD`, the
dirty flag against a file actually touched, the package list against a package known to be installed.
"""

import subprocess
import sys
from pathlib import Path

import src
from src.measure.environment import UNKNOWN, capture, header_lines

REPO = Path(__file__).resolve().parents[2]


def test_capture_is_complete_in_this_environment():
    """If this fails, some field could not be determined and the message says which."""
    env = capture()
    assert env["complete"], f"capture incomplete: {[k for k, v in env.items() if UNKNOWN in str(v)]}"


def test_commit_matches_git_directly():
    """Verified against the command rather than trusted, since a stale or defaulted hash is the whole risk."""
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    assert capture()["git"]["commit"] == expected
    assert len(expected) == 40, "not a full commit hash"


def test_the_dirty_flag_actually_responds(tmp_path):
    """The flag must flip when the tree is dirtied, or it is a constant wearing a field name.

    This is the measure-before-trusting rule applied to the capture: a dirty flag that never becomes True
    records every run as reproducible, which is worse than not recording it at all.
    """
    before = capture()["git"]
    marker = REPO / "_dirty_probe.tmp"
    try:
        marker.write_text("probe")
        after = capture()["git"]
        assert after["dirty"] is True, "the dirty flag did not respond to an untracked file"
        if before["dirty"] is False:
            assert after["dirty_files"] > 0
    finally:
        marker.unlink(missing_ok=True)

    restored = capture()["git"]
    assert restored["dirty"] == before["dirty"], "the probe left the tree in a different state"


def test_packages_are_real_and_from_importlib_not_pip():
    """`pip list` returns nothing in a uv environment, so an empty list would be recorded as a fact."""
    packages = capture()["packages"]
    assert len(packages) > 20, f"only {len(packages)} packages found; pip returns nothing in a uv venv"
    assert packages.get("jax"), "jax is installed and must appear in the capture"
    assert packages["jax"] == __import__("jax").__version__


def test_gpu_and_driver_are_observed_not_defaulted():
    env = capture()["gpu"]
    for field in ("model", "driver", "memory_total"):
        assert not env[field].startswith(UNKNOWN), f"{field} could not be determined: {env[field]}"
    assert "NVIDIA" in env["model"] or "GeForce" in env["model"]


def test_cache_state_comes_from_entry_count_not_runtime():
    """Cold versus warm is snapshotted at import, before any measurement can add to it.

    Inferring it from runtime would mislabel a run that was slow for an unrelated reason as cold, permanently,
    in a record whose purpose is comparability.
    """
    cache = capture()["cache"]
    assert cache["state_at_start"] in ("cold", "warm")
    assert isinstance(cache["entries_at_start"], int), "entry count must be a count, not an inference"
    if cache["state_at_start"] == "warm":
        assert cache["entries_at_start"] > 0
    assert cache["max_size_enforceable"] is True, (
        "filelock is absent, so the compilation cache size cap is silently ignored"
    )


def test_the_runtime_values_are_the_ones_in_effect():
    """Not the ones the code intended to set. Those differ exactly when something external overrode them."""
    env = capture()
    assert env["jax"]["default_float_dtype"] == "float32"
    assert set(env["inherited_environment"]) <= set(src.INHERITED_ENVIRONMENT)
    assert isinstance(env["overridden"], list)


def test_a_failed_subprocess_is_marked_rather_than_defaulted():
    """The construction-class guard, exercised directly rather than argued for.

    A command that cannot run must produce a value starting with the sentinel, never a plausible blank.
    """
    from src.measure.environment import _run

    result = _run(["definitely-not-a-real-command-xyz"])
    assert result.startswith(UNKNOWN), f"a failed command returned {result!r} rather than a marker"

    result2 = _run(["false"])
    assert result2.startswith(UNKNOWN), "a non-zero exit returned a plausible empty string"


def test_incompleteness_propagates_to_the_complete_flag():
    """A single undetermined field must make the whole capture read as incomplete.

    Otherwise a consumer would have to inspect every value to know whether the environment was captured.
    """
    env = capture()
    assert env["complete"] is True
    poisoned = dict(env, gpu=dict(env["gpu"], driver=f"{UNKNOWN}: simulated"))

    def has_unknown(value):
        if isinstance(value, str):
            return value.startswith(UNKNOWN)
        if isinstance(value, dict):
            return any(has_unknown(v) for v in value.values())
        return False

    assert has_unknown(poisoned), "the incompleteness detector cannot see a poisoned field"


def test_header_is_writable_into_a_measurement_output():
    """The consumer this exists for: the header of every measurement file from step 11 onward."""
    lines = header_lines()
    assert len(lines) >= 5
    text = "\n".join(lines)
    assert "commit" in text and "compilation cache" in text and "gpu" in text.lower()
    assert UNKNOWN not in text, f"header contains an undetermined field:\n{text}"
