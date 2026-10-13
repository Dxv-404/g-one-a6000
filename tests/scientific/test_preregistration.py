"""The result-leakage guard. Build step 11.

`docs/analysis-structure.md` commits three things before any Phase B run, with the git history as the
evidence: the tier ordering and that tier one is primary, tier two's feature list, clustering method and
R-squared admission threshold, and tier three's classifier form.

**This guard is kept even though the same values are also protected structurally**, and the two do different
work. The structural version makes the step 13 quantities required arguments, so a caller who has not derived
them cannot obtain a number at all: that PREVENTS a violation. This guard DETECTS one, and it detects a
different violation: a value fixed at the right time but derived from the wrong data, or a pre-registered
choice edited after results were seen. Required arguments cannot see either.

**The check: for every Phase B run in `runs.md`, the commit that last modified the pre-registered files must
be an ancestor of the commit the run executed from.** If the pre-registration was edited after a run, it is
not an ancestor and this fails.

**Not inert while no Phase B run exists.** A guard whose loop body never executes is a guard that cannot fail,
and this project has met enough of those to distrust one on sight. The mechanism is therefore self-tested
against a known pair of commits from this repository's own history, so the machinery is exercised on every
run whether or not there is anything to guard yet.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

#: The artefacts the specification pre-registers. Changing any of them after a Phase B run is result leakage.
#:
#: `descriptors.py` is here for `NAMED_STRATEGIES`, which fixes what tier three can find; `discovery.py` for
#: the candidate list and its ORDER, which stage two makes load-bearing; `classifier.py` for the classifier
#: form; `analysis.py` for the thresholds and the clustering method.
PREREGISTERED = (
    "configs/analysis.py",
    "src/measure/descriptors.py",
    "src/measure/discovery.py",
    "src/measure/classifier.py",
)

#: A run is Phase B if its notes say so. Recorded as a convention in `runs.md`'s field rules rather than
#: inferred from the configuration column, because all four run configurations appear in both phases.
#:
#: **The token is `Phase B: yes` because the previous marker matched its own negation.** It was
#: `re.compile(r"phase\s*b", re.IGNORECASE)`, and every non-Phase-B run in the registry declares itself with
#: the words "NOT Phase B and NOT a characterisation run". All eleven were being SELECTED as Phase B runs.
#:
#: **It stayed green by luck rather than by working.** The guard then checks that each selected run descends
#: from the pre-registration commit, and the pre-registration happened to be older than every registered run,
#: so the ancestry held and the gate passed. It fired the moment a pre-registered file was touched, on runs
#: that are not Phase B at all.
#:
#: **And the skip branch was unreachable.** The docstring below says it skips with a reason when no Phase B
#: run exists, so that a report says "not yet applicable" rather than "checked and fine". That branch has not
#: been taken since the first run was registered, because the marker always matched something.
#:
#: A negative lookbehind for "not " was considered and rejected: it is fragile against "is not a Phase B
#: run" and every other phrasing. A token that cannot be negated by prefixing a word is the structural fix.
PHASE_B_MARKER = re.compile(r"phase\s*b\s*:\s*yes", re.IGNORECASE)


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"git {' '.join(args)} failed:\n{out.stderr}"
    return out.stdout.strip()


def _is_ancestor(older: str, newer: str) -> bool:
    out = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer], capture_output=True, text=True, timeout=60
    )
    return out.returncode == 0


def _rows():
    """Parse `runs.md`'s table. Returns dicts keyed by the header names."""
    text = Path("runs.md").read_text(encoding="utf-8")
    rows = []
    header = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells, strict=True)))
    return rows


def test_the_ancestor_mechanism_actually_works():
    """**Coverage, and it comes first because the guard below has an empty loop until Phase B starts.**

    A gate that cannot fail manufactures confidence. This exercises the machinery on two real commits from
    this repository, so a broken `merge-base` invocation, a wrong argument order, or a `git` that is not
    present fails here rather than silently passing the guard for months.
    """
    head = _git("rev-parse", "HEAD")
    parent = _git("rev-parse", "HEAD~1")
    assert _is_ancestor(parent, head), "HEAD~1 must be an ancestor of HEAD; the mechanism is broken"
    assert not _is_ancestor(head, parent), (
        "HEAD must NOT be an ancestor of HEAD~1. If this passes, the argument order is reversed and the "
        "guard would accept a pre-registration edited after the run"
    )


def test_every_preregistered_file_exists_and_has_a_commit():
    """The guard resolves each file's last-touching commit. A file that has moved or been deleted would make
    that resolution silently empty, and an empty commit compares as an ancestor of nothing."""
    for path in PREREGISTERED:
        assert Path(path).exists(), f"pre-registered file missing: {path}"
        sha = _git("log", "-1", "--format=%H", "--", path)
        assert re.fullmatch(r"[0-9a-f]{40}", sha), f"no commit found for {path}: {sha!r}"


def test_the_phase_b_marker_does_not_match_its_own_negation():
    """**The defect that made this necessary was in this file.**

    The marker was `phase\\s*b`, which matches "NOT Phase B". Every run in the registry that declares itself
    NOT Phase B was selected as one, and the gate passed anyway because the pre-registration commit happened
    to predate every run. A guard that is green for the wrong reason is worse than a red one.

    Both directions are asserted, because a marker that matches nothing would also pass a
    does-not-match-the-negation test.
    """
    positives = ["Phase B: yes", "phase b: yes", "Phase B : Yes", "characterisation sweep. Phase B: yes"]
    negatives = [
        "NOT Phase B and NOT a characterisation run",
        "not phase b",
        "this is not a Phase B run",
        "Phase A only",
        "",
    ]
    for text in positives:
        assert PHASE_B_MARKER.search(text), f"marker failed to match an affirmative declaration: {text!r}"
    for text in negatives:
        assert not PHASE_B_MARKER.search(text), f"marker matched a non-Phase-B note: {text!r}"


def test_every_registered_run_declares_its_phase_unambiguously():
    """A run that mentions Phase B at all must do so in the affirmative token or in a clear negation.

    Guards the convention itself: a note reading "Phase B?" or "possibly Phase B" would be selected by
    neither branch and would silently escape the guard.
    """
    import re as _re

    unclear = []
    for row in _rows():
        notes = row.get("notes", "")
        if not _re.search(r"phase\s*b", notes, _re.IGNORECASE):
            continue
        if PHASE_B_MARKER.search(notes):
            continue
        if _re.search(r"\bnot\s+(a\s+)?phase\s*b", notes, _re.IGNORECASE):
            continue
        unclear.append(row.get("run_id"))
    assert not unclear, (
        f"runs {unclear} mention Phase B without declaring it as `Phase B: yes` or negating it clearly. "
        "Such a run is selected by neither branch and escapes the guard entirely."
    )


def test_no_phase_b_run_predates_its_pre_registration():
    """**The guard.** For every Phase B run, the pre-registration must already have been committed.

    Skips with a reason rather than passing silently when no Phase B run exists yet, so the run report says
    "not yet applicable" instead of "checked and fine". The two read identically in a green suite and mean
    entirely different things.
    """
    rows = [r for r in _rows() if PHASE_B_MARKER.search(r.get("notes", ""))]
    if not rows:
        pytest.skip("no Phase B run registered yet; the mechanism is exercised by the test above")

    prereg = max(
        (_git("log", "-1", "--format=%ct", "--", p), _git("log", "-1", "--format=%H", "--", p))
        for p in PREREGISTERED
    )[1]

    for row in rows:
        sha = row.get("git_sha", "").strip()
        assert sha, f"Phase B run {row.get('run_id')} has no git_sha; it cannot be checked or reproduced"
        assert _is_ancestor(prereg, sha), (
            f"Phase B run {row.get('run_id')} executed from {sha}, which does not descend from the "
            f"pre-registration commit {prereg}. The pre-registered feature list, thresholds, clustering "
            "method or classifier form was changed after that run, which is result leakage: the analysis "
            "was chosen with the results visible."
        )


def test_the_marker_convention_is_documented():
    """The guard selects Phase B runs by a marker in the notes column. A convention nobody wrote down is one
    the next person will not follow, and a run that silently fails to match is a run this guard never
    checks."""
    text = Path("runs.md").read_text(encoding="utf-8")
    assert PHASE_B_MARKER.search(text), (
        "runs.md does not document the Phase B marker convention, so a future run could omit it and slip "
        "past the leakage guard without anything failing"
    )
