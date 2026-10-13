"""Gate E1. Determinism across restarts. Build step 10.

The build spec's definition:

> Same seed reproduces discrete state exactly, positions, actions, alive flags, scores, death causes, across
> restarts. Float pools reproduce within a stated tolerance. Cross-device bitwise equality is not required
> and is not achievable for float GPU work.

**The stated tolerance is zero, and it is measured rather than chosen.** Two separate processes, the second
forced to recompile from a cold private compilation cache, agree bitwise on every field including the float
pools: maximum absolute difference 0.000e+00 across energy, stamina, health and every accumulator.

**So this is asserted as an identity, not as a threshold.** CLAUDE.md prefers that wherever it is available:
an identity has no strictness to choose, cannot be quietly loosened, and does not go stale when the honest
number moves. If it ever fails, the response is to find out what became nondeterministic, not to introduce a
tolerance. A tolerance here would be a number with no provenance guarding a property that currently holds
exactly.

**Two things make this stronger than a naive same-process check, and both are asserted rather than assumed.**

    a real restart      separate subprocesses, so module-level state, cache warmth and PRNG global state
                        cannot carry over. A same-process rerun would pass against almost any bug
    a real recompile    the second process gets an empty private cache directory, so XLA compiles from
                        scratch. Without this the test compares one executable against ITSELF loaded from
                        disk, which is a much weaker claim than it appears to make. The number of cache
                        entries written is asserted to be non-zero for exactly this reason
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SEED = 0


def _run(tmp: Path, out: Path, cold_cache: Path | None) -> int:
    """Run one episode in a fresh interpreter. Returns the number of cache files the run produced."""
    env = dict(os.environ)
    if cold_cache is not None:
        if cold_cache.exists():
            shutil.rmtree(cold_cache)
        cold_cache.mkdir(parents=True)
        env["JAX_COMPILATION_CACHE_DIR"] = str(cold_cache)
    proc = subprocess.run(
        [sys.executable, "-m", "tests.engineering._e1_replay", str(SEED), str(out)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, f"the replay subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return len(list(cold_cache.rglob("*"))) if cold_cache is not None else 0


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e1")
    a, b = tmp / "a.npz", tmp / "b.npz"
    _run(tmp, a, None)
    written = _run(tmp, b, tmp / "coldcache")
    return np.load(a), np.load(b), written


def test_the_second_run_actually_recompiled(two_runs):
    """**Coverage, and it comes first because everything below is weaker without it.**

    With a warm shared cache the second process loads the same executables the first one compiled, so the
    comparison degenerates into checking that one binary is deterministic. That is true and nearly
    uninteresting. The claim E1 makes is that a fresh compilation reproduces the same numbers.
    """
    _, _, written = two_runs
    assert written > 0, (
        "the cold cache directory is empty, so the second run compiled nothing and must have loaded "
        "executables from somewhere else. This test is then comparing an executable against itself and is "
        "far weaker than it reads."
    )
    print(f"\n  E1: the cold run wrote {written} compilation cache entries")


def test_the_episode_exercises_what_it_claims_to_compare(two_runs):
    """The other half of coverage. Comparing two episodes in which nothing happened would agree perfectly
    and prove nothing, which is the value-level form of an inert assertion."""
    a, _, _ = two_runs
    actions = a["traj.action_emitted"]
    assert len(np.unique(actions)) > 1, f"only action {np.unique(actions)} was ever emitted"
    assert a["traj.prey_moved"].any(), "no prey moved at any tick, so positions could not have diverged"
    assert (a["float.prey_energy"] != a["float.prey_energy"][0]).any() or float(
        np.max(a["float.prey_energy"])
    ) > 0, "the float pools carry no variation to compare"


def test_discrete_state_reproduces_exactly(two_runs):
    """The spec's requirement. Positions, actions, alive flags, scores, death causes."""
    a, b, _ = two_runs
    fields = [k for k in a.files if a[k].dtype.kind in "biu"]
    assert fields, "no discrete fields were dumped; the replay module has changed"
    for k in sorted(fields):
        n = int(np.sum(a[k] != b[k]))
        assert n == 0, (
            f"{k} differs in {n} of {a[k].size} elements between two runs at the same seed. Discrete state "
            "must reproduce exactly across restarts, and it is what every death cause, action log and score "
            "in the characterisation is built from."
        )


def test_float_pools_reproduce_exactly_and_the_stated_tolerance_is_zero(two_runs):
    """**Stricter than the spec requires, because the measurement permits it.**

    The spec allows a tolerance for float pools. Measured across a genuine recompilation the difference is
    exactly zero, so the honest gate is equality. Read the module docstring before changing this: the
    response to a failure here is to find the nondeterminism, not to add a tolerance.
    """
    a, b, _ = two_runs
    fields = [k for k in a.files if a[k].dtype.kind == "f"]
    assert fields, "no float fields were dumped; the replay module has changed"
    worst, where = 0.0, ""
    for k in sorted(fields):
        d = float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) if a[k].size else 0.0
        if d > worst:
            worst, where = d, k
    print(f"  E1: worst float difference {worst:.3e}" + (f" in {where}" if where else ""))
    assert worst == 0.0, (
        f"float state differs by {worst:.3e} in {where} between two runs at the same seed. This has always "
        "been exactly zero on this device. Do not add a tolerance: establish what changed, because a float "
        "divergence large enough to flip an argmax becomes a discrete divergence in the next tick."
    )


def test_every_accumulator_is_covered_by_this_gate(two_runs):
    """**A ratchet.** An accumulator added later but not dumped would sit outside E1's reach, and nothing
    about a green run would say so. Asserted as set equality rather than containment, so a dumped field that
    no longer exists also fails.

    This is the same shape as the design audit's parameter ratchet, and it exists for the same reason: the
    gate's coverage falls with every build step while its name and reputation stay constant, unless
    something asserts the coverage directly.
    """
    from src.env.env import Accumulators

    a, _, _ = two_runs
    dumped = {k.split(".", 1)[1] for k in a.files if k.startswith("acc.")}
    expected = set(Accumulators._fields)
    assert dumped == expected, (
        f"accumulators not covered by E1: {sorted(expected - dumped)}; "
        f"dumped fields that are no longer accumulators: {sorted(dumped - expected)}"
    )
