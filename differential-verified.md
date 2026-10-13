# differential-verified.md

The full differential tier, recorded per commit that a registered run executed from.

## Why this file exists

**The six heaviest differential functions are excluded from the pre-push hook, and they are the strongest
gates in this project.** They are the tick-by-tick oracle comparisons: a full episode, every pool, every ray,
compared against an independently written NumPy implementation. Eleven minutes is too long for a hook, and a
hook that slow gets bypassed within a week, which is the failure this repository has already watched twice.

**So they bind to RUN REGISTRATION instead, which is where their result actually matters.** A registered run
is evidence. A number in `runs.md` that anyone might later cite should have been produced by a simulator
whose oracle comparison passed at that commit. A number produced on every push, where nobody reads it, buys
less and costs more.

This is the same shape as `citations-verified.md`: a claim that something was checked, recorded with what was
checked and when, rather than asserted in prose.

## The rule

**Before registering a run in `runs.md`, run the full differential tier at the run's commit and record the
result here.** `tests/scientific/test_differential_binding.py` asserts that every run from run 14 onward has
a row here whose commit matches the run's `git_sha`.

```
uv run pytest tests/differential -q
```

**Runs 1 to 13 are exempt and the exemption is stated rather than silent.** They predate this rule. Applying
it retroactively would mean either fabricating verification that never happened or marking thirteen genuine
runs as unverifiable, and neither is honest. The cutoff is recorded in the test so it cannot drift.

## What a row means

`commit` is the git sha the differential ran at. `result` is `pass` or `fail` with the counts. A run whose
differential FAILED is still registered, with the failure recorded, because the reporting standard requires
every run to be reported including failures. It is the interpretation that changes, not the disclosure.

## Verifications

| commit | date | result | notes |
|---|---|---|---|
| 15f6c36 | 2026-08-26 | pass, 96 passed | Full tier, 667s. The last commit before this rule existed. Recorded as the baseline so run 14 onward has something to match against |
| c836ed7 | 2026-08-28 | pass, 96 passed | Full tier, 761s, clean tree. Verified BEFORE registering runs 14 to 18, which is the workflow this file exists for and its first real exercise. The six heaviest oracle comparisons are excluded from the pre-push hook, so this is the only place they bind |
| 53cb730 | 2026-08-28 | pass, 96 passed | Full tier, 665s, clean tree. Verified before the M4 characterisation sweep, which refuses to start without a recorded pass at its cited commit. Also the post-change check for the r_pred decoupling: that change lands identically in `configs/default.py` for both the implementation and the oracle, so the differential is INERT for it per D-041 and this pass is not evidence about it. The prediction table in the commit message is the evidence |
| 055e823 | 2026-08-28 | pass, 96 passed | Full tier, 656s, clean tree, run inside the battery pipeline BEFORE batteries B, C and D executed. Covers runs 19 to 21 |
| 3c0fead | 2026-08-29 | pass, 96 passed | Full tier, 621s, clean tree. The D-046 commit: five near-conditioned accumulators and the descriptor re-basing. The differential is INERT for the accumulator change itself, which landed identically on both sides per D-041; the evidence for that change is the prediction record in D-046, and this pass covers everything else plus the 26-field set equality. Bound the first battery attempt on the re-based stack, which failed for the two reasons D-048 records |
| 7565e0a | 2026-08-29 | pass, 96 passed | Full tier, 636s, clean tree. The D-048 commit: `away_near` and `away_far`, the emitted-direction flee event. INERT for that change for the same D-041 reason, with the prediction record in D-048; the 28-field set equality is what this pass adds. Covers run 22 |
| 414ea1a | 2026-08-31 | pass, 96 passed | Full tier, 625s, clean tree. The gate-queue commit: the resume schema guard, the pairing assertion, and the two tie-break measurement scripts. **The differential is not evidence about the guard**, which is a runner-level precondition the oracle does not model; its evidence is the eight tests, including a stale-checkpoint fixture and a can-it-fail check on the pairing assertion. Covers runs 23 and 24 |
| ba64e6c | 2026-08-31 | pass, 96 passed | Full tier, 640s, clean tree. The detector-diagnosis commit. **Measurement scripts only, so the differential is not evidence about the diagnoses**; it establishes that the simulator they measured is the one the oracle agrees with, which is the property a diagnosis needs and the only one this pass supplies. Covers runs 25 and 26 |
| ce54c34 | 2026-08-31 | pass, 96 passed | Full tier, 597s, clean tree. **The first full tier since the oracle CHANGED**: queue item E extended the oracle's record from four fields to nine, so this is the pass that establishes the extension is correct rather than merely present, and the record's field sets are now asserted equal on every tick. Covers run 27 |
| 82cfd8e | 2026-09-06 | pass, 96 passed | Full tier, 625s, clean tree. The commit that registered runs 28 and 29 and their two drivers. **Executed here because the pre-push binding gate refused the push**: runs 28 and 29 cite 77daa26, which had no recorded pass. Nothing under `src/`, `tests/` or `configs/` differs between ce54c34, 77daa26 and this commit (`git diff --stat` is empty for all three paths), so this pass is the same program the ce54c34 pass ran, re-executed rather than inferred |
| 77daa26 | 2026-09-06 | pass, 96 passed | The commit runs 28 and 29 cite. **Not executed at this commit**: the tier ran at 82cfd8e, whose `src/`, `tests/` and `configs/` are byte-identical to this commit's, as the row above records. Entered so the binding gate matches the runs to a pass on the code they measured, and stated this way so the record says what happened rather than what would have happened |
| cfec9ba | 2026-09-06 | pass, 96 passed | Full tier, 594s, clean tree. The M4 commit, D-059: the runner keeps final populations, runs the null in-line and checkpoints per seed. **Measurement-runner changes only, so the differential is not evidence about them**; their evidence is the 17 runner tests. This pass establishes the simulator the null baseline is about to run on is the one the oracle agrees with. Covers run 32 |
| 91a67c5 | 2026-09-06 | pass, 96 passed | Full tier, 605s, clean tree. The S5 and S6 commit, D-060: src/measure/power.py and its tests. **Nothing under src/env changed**, so the differential is not evidence about the power module; its evidence is its nine tests and the S6 curve reproducing to three decimals after the vectorisation. This pass binds the treatment baseline, run 33 |
| 8a7dd00 | 2026-09-06 | pass, 96 passed | Full tier, 670s, clean tree. The dashboard-brief commit: replay generator, viewer, corpus 099 and 100, runs 35 to 38. **Measurement and presentation code only**; nothing under src/env, src/reference or configs changed since 91a67c5, which git diff --stat confirms empty for those paths. Executed here because the binding gate refused the push for runs 35 to 38 |
| 2b38349 | 2026-09-06 | pass, 96 passed | Full tier, 670s, clean tree. The commit runs 35 to 38 cite. **Not executed at this commit**: the tier ran at 8a7dd00, and src/env, src/reference and configs are byte-identical between 91a67c5, this commit and 8a7dd00. This commit changed src/measure/power.py and its tests, which the differential does not exercise. Stated so the record says what happened |
| e6b93b7 | 2026-10-13 | pass, 96 passed | Full tier, 844s, clean tree, RTX 4060 laptop. The D-061 and D-062 commit: sweep progress reporting and health checks, and the A6000 repository builder. **Measurement-runner and test code only**; nothing under src, src/reference or configs changed. Engineering 358 passed (334 + 24), scientific 59 passed with S6 failing as known under Q-051, 1 skipped, 1 xpassed. The derived repository for the A6000 ships this runnable code and names this commit as its verified commit |
