"""CLAUDE.md section 7's repository map, checked against the filesystem.

**A map is a stated fact with no consumer, and this one had drifted in both directions at once.** It named
`measure/learnability` and `measure/throughput`, neither of which exists, and omitted `env/observation`,
`env/env`, `evo/genome`, and four of the five modules under `measure/`.

**It had a real cost rather than a theoretical one.** Reasoning about `n_eng_min` during the session 14
analysis audit began from the assumption that `src/measure/learnability.py` existed, because the map says so.
It does not; learnability arrives at step 12 with M4, so `n_eng_min` has no implementation and its
aggregation level is not pinned anywhere. The map made an absent module look present, which is the specific
way a stale map misleads: it is consulted precisely when someone does not already know what is there.

CLAUDE.md's own rule is that the fix for a drifting fact is not to correct it but to give it something that
breaks when it stops being true. This is that something.

**Deliberately checks module NAMES and not descriptions.** The trailing prose on each line is commentary and
should stay free; pinning it would make the map annoying to edit and the check would be worked around. What
must not drift is the set of modules, which is exactly the part a reader relies on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Packages under `src/` that the map is required to describe. A new package added here without a map entry
#: fails, which is the direction that matters: the map going silently incomplete is how it drifted before.
PACKAGES = ("env", "agent", "evo", "measure", "render", "reference")


def _map_block() -> str:
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    m = re.search(r"^## 7\. Repository map\s*\n\s*```\n(.*?)^```", text, re.S | re.M)
    assert m, "CLAUDE.md section 7 has no fenced repository map. It is the thing this test exists to check."
    return m.group(1)


def _mapped_modules(block: str, package: str) -> set[str]:
    m = re.search(rf"^\s*{re.escape(package)}/\s+(.*)$", block, re.M)
    assert m, f"the repository map has no line for `src/{package}/`"
    # Names are comma-separated; anything after two or more spaces is free commentary.
    names = re.split(r"\s{2,}", m.group(1).strip())[0]
    return {n.strip() for n in names.split(",") if n.strip()}


def _real_modules(package: str) -> set[str]:
    d = ROOT / "src" / package
    assert d.is_dir(), f"src/{package}/ does not exist but the map names it"
    return {p.stem for p in d.glob("*.py") if p.stem != "__init__"}


@pytest.mark.parametrize("package", PACKAGES)
def test_the_repository_map_names_exactly_the_modules_that_exist(package: str) -> None:
    """Both directions, because the map drifted in both.

    Naming a module that does not exist is the worse half: it makes an absent thing look present to a reader
    who is consulting the map precisely because they do not know. Omitting one is milder and still wrong.
    """
    mapped = _mapped_modules(_map_block(), package)
    real = _real_modules(package)
    phantom = sorted(mapped - real)
    missing = sorted(real - mapped)
    assert not phantom, (
        f"CLAUDE.md's map names src/{package}/ modules that do not exist: {phantom}. A map that names an "
        "absent module is worse than one that omits a present one, because it is consulted by people who do "
        "not already know."
    )
    assert not missing, (
        f"CLAUDE.md's map omits src/{package}/ modules that exist: {missing}. Add them, or the map goes "
        "quietly incomplete, which is how it drifted before."
    )


def test_every_src_package_appears_in_the_map() -> None:
    """A new package added to `src/` without a map line fails here rather than being noticed years later."""
    block = _map_block()
    on_disk = {p.name for p in (ROOT / "src").iterdir() if p.is_dir() and not p.name.startswith("__")}
    unmapped = sorted(d for d in on_disk if not re.search(rf"^\s*{re.escape(d)}/", block, re.M))
    assert not unmapped, f"packages under src/ with no line in CLAUDE.md's repository map: {unmapped}"


def test_the_map_can_fail() -> None:
    """**Confirm the parser would actually notice a wrong name**, rather than passing on anything.

    CLAUDE.md requires that a new gate be shown capable of failing. Doing it inside the test keeps the
    demonstration attached to the assertion rather than living in a log entry nobody re-runs, and it costs
    nothing because the check is pure string work.
    """
    block = _map_block()
    real = _real_modules("evo")
    assert _mapped_modules(block, "evo") == real, "precondition: the map is correct before this is perturbed"

    broken = block.replace("evo/", "evo/       phantom_module,", 1)
    mapped = {
        n.strip()
        for n in re.split(r"\s{2,}", re.search(r"^\s*evo/\s+(.*)$", broken, re.M).group(1).strip())[0].split(
            ","
        )
        if n.strip()
    }
    assert "phantom_module" in mapped - real, (
        "the parser did not pick up an injected phantom module, so a real one would pass unnoticed"
    )
