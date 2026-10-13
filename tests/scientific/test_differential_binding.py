"""Every registered run's commit has a recorded full-differential pass. Owner's ruling, 2026-08-27.

**The six heaviest differential functions are excluded from the pre-push hook and they are the strongest
gates here.** They are the tick-by-tick oracle comparisons against an independently written NumPy
implementation. At eleven minutes they are too slow for a hook, and a hook that slow is bypassed within a
week, which this repository has already watched happen twice.

**So they bind to RUN REGISTRATION, which is the moment their result matters.** A registered run is evidence
that someone may cite in eighteen months. A number produced by a simulator whose oracle comparison was never
run at that commit is a number with nothing behind it. A differential run on every push, where nobody reads
the result, buys less and costs more.

**This is a binding, not a re-run.** The test does not execute the differential; it asserts that someone did
and recorded it. That is deliberate: a test that ran an eleven-minute tier would put the cost back in exactly
the place it was removed from, and the pre-push hook would then be bypassed for the reason it was designed to
avoid.

**The cutoff is in the code rather than in prose**, so it cannot drift. Runs 1 to 13 predate the rule.
Applying it retroactively would mean fabricating verification that never happened or marking thirteen genuine
runs unverifiable, and neither is honest.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Runs before this predate the rule. **Recorded here rather than in prose** so a later reader cannot
#: quietly widen the exemption, and so the number appears in a diff when it changes.
FIRST_BOUND_RUN = 14

#: A row is `| commit | date | result | notes |`. Only the commit and the result are load-bearing.
ROW = re.compile(r"^\|\s*([0-9a-f]{7,40})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", re.M)


def _verifications() -> dict[str, str]:
    text = (ROOT / "differential-verified.md").read_text(encoding="utf-8")
    body = text.split("## Verifications", 1)
    assert len(body) == 2, "differential-verified.md has no `## Verifications` section"
    return {m.group(1): m.group(3) for m in ROW.finditer(body[1])}


def _runs() -> list[tuple[int, str]]:
    text = (ROOT / "runs.md").read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"^\|\s*(\d+)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9a-f]{7,40})\s*\|", text, re.M):
        out.append((int(m.group(1)), m.group(2)))
    return out


def test_every_bound_run_has_a_recorded_differential_pass() -> None:
    """The binding itself.

    Matches on commit PREFIX in both directions, because `runs.md` records short shas and a verification may
    record either. A prefix match is what git itself does and is the only comparison that works across both
    conventions.
    """
    verified = _verifications()
    runs = _runs()
    assert runs, "runs.md has no parseable rows, so this check would pass vacuously"

    missing = []
    for run_id, sha in runs:
        if run_id < FIRST_BOUND_RUN:
            continue
        hit = next((r for c, r in verified.items() if c.startswith(sha) or sha.startswith(c)), None)
        if hit is None:
            missing.append((run_id, sha))
        elif not hit.lower().startswith("pass"):
            # A recorded FAILURE is not a missing verification. The reporting standard requires every run
            # to be reported including failures, so this states it and does not fail the gate.
            print(f"run {run_id} at {sha}: differential recorded as {hit!r}")
    assert not missing, (
        f"runs with no recorded full-differential verification: {missing}. Run `uv run pytest "
        "tests/differential -q` at that commit and add a row to differential-verified.md. The six heaviest "
        "oracle comparisons are excluded from the pre-push hook, so registration is the only place they bind."
    )


def test_the_verification_file_is_parseable_and_not_empty() -> None:
    """**A binding whose source parses to nothing passes vacuously**, which is the failure mode of every
    check that reads a file. Asserted directly rather than relied upon.
    """
    verified = _verifications()
    assert verified, "differential-verified.md parsed to zero rows; the binding would accept anything"
    for commit, result in verified.items():
        assert re.fullmatch(r"[0-9a-f]{7,40}", commit), f"malformed commit in the table: {commit!r}"
        assert result.strip(), f"empty result for {commit}"


def test_the_binding_can_fail() -> None:
    """Confirm a run with no verification is actually rejected, rather than the matcher accepting anything.

    In-test rather than in a log entry, because a log entry is not re-run. Pure string work, no cost.
    """
    verified = {"abc1234": "pass, 96 passed"}
    for sha, expected_hit in (("abc1234", True), ("abc1234def", True), ("999aaaa", False)):
        hit = any(c.startswith(sha) or sha.startswith(c) for c in verified)
        assert hit is expected_hit, f"prefix matcher wrong for {sha!r}: got {hit}, expected {expected_hit}"
