"""The power simulation's own correctness, so gate S6 rests on arithmetic that has been shown to fail.

Gate S6 reads `src.measure.power` and asserts a number against 0.8. These tests check the pieces that number
is built from, at both ends of their range and in the interior, because a detection rate is a symptom with
more than one sufficient cause: a broken bootstrap that always excludes zero reports perfect power.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.measure import power as P


def test_the_interquartile_mean_drops_both_tails():
    x = np.array([100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -100.0])
    # trim_mean cuts floor(0.25 n) = 2 values from each end of ten, leaving the middle six
    assert P.iqm(x) == pytest.approx(np.mean([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert P.iqm(np.full(8, 3.5)) == 3.5


def test_the_row_wise_interquartile_mean_agrees_with_trim_mean():
    """The bootstrap uses the vectorised form; the reported statistic uses scipy. One definition, checked
    at odd and even lengths, where int(0.25 n) rounds differently."""
    rng = np.random.default_rng(11)
    for n in (7, 8, 10, 13, 40):
        m = rng.normal(size=(6, n))
        assert np.allclose(P.iqm_rows(m), [P.iqm(row) for row in m])


def test_the_bootstrap_interval_brackets_the_statistic_and_narrows_with_more_runs():
    rng = np.random.default_rng(0)
    small = rng.normal(0.0, 1.0, 8)
    large = rng.normal(0.0, 1.0, 64)
    lo_s, hi_s = P.bootstrap_interval(small, np.random.default_rng(1))
    lo_l, hi_l = P.bootstrap_interval(large, np.random.default_rng(1))
    assert lo_s <= P.iqm(small) <= hi_s
    assert lo_l <= P.iqm(large) <= hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s), "an interval over eight times the runs must be narrower"


def test_no_effect_is_detected_at_about_the_nominal_rate():
    """d = 0: the interval should exclude zero about 2.5 percent of the time on the upper side. Asserted
    loosely because the percentile bootstrap on ten runs is not exact, and tightly enough that a bootstrap
    which always excludes zero cannot pass."""
    null = np.random.default_rng(2).normal(0.3, 0.1, 10)
    rate = P.detection_rate(null, 10, d=0.0, n_sims=400, n_boot=200, rng=np.random.default_rng(3))
    assert rate < 0.15, f"a null effect was 'detected' {rate:.2f} of the time"


def test_a_huge_effect_is_always_detected():
    null = np.random.default_rng(4).normal(0.3, 0.1, 10)
    rate = P.detection_rate(null, 10, d=8.0, n_sims=200, n_boot=200, rng=np.random.default_rng(5))
    assert rate == 1.0


def test_power_rises_with_seed_count_in_the_interior():
    """The interior point the rule in CLAUDE.md asks for: not only d = 0 and d = huge, but a moderate
    effect whose detection must increase with runs. Monotone within simulation noise."""
    null = np.random.default_rng(6).normal(0.3, 0.1, 10)
    curve = P.power_curve(null, (4, 10, 30), d=1.0, n_sims=300, n_boot=200, rng=np.random.default_rng(7))
    assert curve[4] < curve[30], curve
    assert 0.0 < curve[10] < 1.0, (
        f"a d = 1 effect at ten runs should be neither hopeless nor certain: {curve}"
    )


def test_a_degenerate_null_is_refused_rather_than_reported():
    with pytest.raises(ValueError, match="no spread"):
        P.detection_rate(np.full(10, 0.3), 10, rng=np.random.default_rng(0))


def test_the_rate_is_reproducible_from_its_generator():
    null = np.random.default_rng(8).normal(0.3, 0.1, 10)
    a = P.detection_rate(null, 10, n_sims=100, n_boot=100, rng=np.random.default_rng(9))
    b = P.detection_rate(null, 10, n_sims=100, n_boot=100, rng=np.random.default_rng(9))
    assert a == b


def test_the_spec_constants_are_the_spec_constants():
    """Determined, not chosen: S6 names d = 1.0 and 80 percent, the canonical names 95 percent."""
    assert P.SMALLEST_EFFECT_D == 1.0
    assert P.REQUIRED_POWER == 0.80
    assert P.CONFIDENCE == 0.95


def test_paired_power_equals_independent_power_when_the_arms_are_uncorrelated():
    """The two functions must agree where their assumptions coincide: differences of two independent draws
    from the null have exactly the noise `detection_rate` simulates. Loose because both are simulations."""
    rng = np.random.default_rng(20)
    null = rng.normal(0.3, 0.1, 10)
    shift = 1.0 * null.std(ddof=1)
    diffs = rng.normal(0.3, 0.1, 10) - rng.normal(0.3, 0.1, 10)
    # give the differences the null's own spread times root two, which is what independence implies
    diffs = (diffs - diffs.mean()) / diffs.std(ddof=1) * (np.sqrt(2.0) * null.std(ddof=1))
    a = P.detection_rate(null, 10, d=1.0, n_sims=400, n_boot=200, rng=np.random.default_rng(21))
    b = P.detection_rate_from_differences(
        diffs, shift, 10, n_sims=400, n_boot=200, rng=np.random.default_rng(22)
    )
    assert abs(a - b) < 0.15, (a, b)


def test_paired_power_rises_as_the_paired_noise_shrinks():
    """The point of pairing: the same effect against smaller difference noise is detected more often."""
    rng = np.random.default_rng(23)
    null = rng.normal(0.3, 0.1, 10)
    shift = 1.0 * null.std(ddof=1)
    # the shift is about 0.1; wide noise five times that, narrow noise equal to it. The first fixture
    # used 1.4 times and 0.56 times, both of which detect at 1.0 on ten runs, so the test could not
    # see the ordering it asserts. A fixture has to be able to show the assertion failing
    wide = rng.normal(0.0, 0.5, 10)
    narrow = wide * 0.2
    a = P.detection_rate_from_differences(
        wide, shift, 10, n_sims=300, n_boot=200, rng=np.random.default_rng(24)
    )
    b = P.detection_rate_from_differences(
        narrow, shift, 10, n_sims=300, n_boot=200, rng=np.random.default_rng(24)
    )
    assert a < 0.9 and b > a, (a, b)  # the wide case must leave room to rise


def test_paired_power_removes_a_real_effect_before_resampling():
    """The observed differences may carry a true effect; the noise is what is resampled, so adding a
    constant to every difference must not change the rate."""
    rng = np.random.default_rng(25)
    diffs = rng.normal(0.0, 0.05, 10)
    a = P.detection_rate_from_differences(
        diffs, 0.03, 10, n_sims=200, n_boot=200, rng=np.random.default_rng(26)
    )
    b = P.detection_rate_from_differences(
        diffs + 5.0, 0.03, 10, n_sims=200, n_boot=200, rng=np.random.default_rng(26)
    )
    assert a == b


def test_paired_power_refuses_constant_differences():
    with pytest.raises(ValueError, match="no spread"):
        P.detection_rate_from_differences(np.full(10, 0.2), 0.01, 10, rng=np.random.default_rng(0))
