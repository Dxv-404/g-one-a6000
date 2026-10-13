"""Gate S7. Reproducibility from config. Build step 10.

The build spec's definition:

> A run reconstructs from its config hash and seed alone, to the E1 standard.

**The interface is the claim.** `tests/scientific/_s7_run.py` takes a config name and a seed and nothing
else. Anything a run needed beyond those two would have to appear in that signature, so its absence is what
makes the claim checkable instead of asserted. A test that re-typed the parameters into both runs would pass
while the property failed.

**To the E1 standard means exactly**, because E1 measured the float difference across a genuine recompilation
at 0.000e+00 and asserts equality rather than a tolerance. A weaker standard here than there would be odd:
an evolutionary run is a composition of episodes, and if each episode reproduces bitwise the composition
does too unless the loop itself introduces nondeterminism, which is the thing being tested.

**Three ways this gate could be green and meaningless, and all three are asserted against.**

    the run is trivial      a run where nothing varies reproduces perfectly. So a different seed must
                            produce a different result, checked directly
    the hash is inert       a hash that never changes agrees with itself. So a different config must
                            produce a different hash
    the hash is salted      Python's `hash` is randomised per process. A hash built on it would differ
                            between the two runs, so the hash is compared ACROSS processes, not within one
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.scientific import _s7_run as runner

SEED = 3


def _run(name: str, seed: int, out: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tests.scientific._s7_run", name, str(seed), str(out)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=1800,
    )
    assert proc.returncode == 0, f"the run subprocess failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s7")
    paths = {}
    for label, (name, seed) in {
        "a": ("s7-small", SEED),
        "b": ("s7-small", SEED),
        "other_seed": ("s7-small", SEED + 1),
        "other_config": ("s7-alt", SEED),
    }.items():
        paths[label] = tmp / f"{label}.npz"
        _run(name, seed, paths[label])
    return {k: np.load(v) for k, v in paths.items()}


def test_the_run_is_not_trivial(runs):
    """**Coverage, first.** A run in which nothing happens reproduces perfectly and demonstrates nothing.

    The fitness history must contain more than one distinct value across the population, or every genome is
    identical and the reproduction claim is about a constant.
    """
    a = runs["a"]
    pop = a["final_population"]
    assert pop.ndim == 2 and pop.shape[0] > 1, f"population shape {pop.shape} is not a population"
    spread = float(np.max(np.std(pop, axis=0)))
    print(f"\n  S7: max per-gene spread across the final population {spread:.6f}")
    assert spread > 0.0, (
        "every genome in the final population is identical, so the run collapsed and reproducing it says "
        "nothing about the loop"
    )


def test_the_same_config_and_seed_reproduce_exactly(runs):
    """The gate. Two separate processes, same config name and seed, compared to the E1 standard."""
    a, b = runs["a"], runs["b"]
    assert sorted(a.files) == sorted(b.files)
    for k in sorted(a.files):
        x, y = a[k], b[k]
        if x.dtype.kind in "US":
            assert str(x) == str(y), f"{k} differs: {x} against {y}"
            continue
        d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        assert d == 0.0, (
            f"{k} differs by {d:.3e} between two runs of the same config and seed. A run that does not "
            "reconstruct from its config and seed cannot be replicated by anyone, which under the reporting "
            "standard means it cannot be published."
        )


def test_the_config_hash_is_stable_across_processes(runs):
    """**Compared across processes on purpose.** Python's `hash` is salted per interpreter unless
    PYTHONHASHSEED is set, so a hash built on it agrees with itself inside one process and disagrees between
    two. Comparing within a process would pass against exactly the implementation this guards."""
    a, b = runs["a"], runs["b"]
    assert str(a["config_hash"]) == str(b["config_hash"]), (
        f"the config hash differs between processes: {a['config_hash']} against {b['config_hash']}. It is "
        "probably built on Python's salted hash rather than on hashlib."
    )
    print(f"  S7: config hash stable at {a['config_hash']}")


def test_a_different_seed_gives_a_different_run(runs):
    """The complement to reproduction. Without it the gate is satisfied by a loop that ignores its seed."""
    a, other = runs["a"], runs["other_seed"]
    assert not np.array_equal(a["final_population"], other["final_population"]), (
        "changing the seed changed nothing, so the run does not depend on its seed and reproducing it is "
        "vacuous"
    )
    assert str(a["config_hash"]) == str(other["config_hash"]), (
        "the config hash changed when only the seed changed; the hash is supposed to identify the CONFIG"
    )


def test_a_different_config_gives_a_different_hash(runs):
    """A hash that does not respond to the config cannot identify a run, and every row in the run registry
    is keyed on it."""
    a, other = runs["a"], runs["other_config"]
    assert str(a["config_hash"]) != str(other["config_hash"]), (
        f"two different configurations hash to the same value {a['config_hash']}. The run registry keys on "
        "this, so runs with different parameters would be indistinguishable in the record."
    )


def test_the_runner_takes_only_a_config_name_and_a_seed():
    """**The interface is the claim.** If a run needed anything else to reproduce, it would have to be a
    parameter here. Asserted so that adding one is a deliberate act with a visible cost."""
    import inspect

    params = list(inspect.signature(runner.run).parameters)
    assert params == ["name", "seed"], (
        f"the runner takes {params}. S7 claims a run reconstructs from its config and seed ALONE, and every "
        "extra parameter is something a replicator would have to be told separately."
    )
