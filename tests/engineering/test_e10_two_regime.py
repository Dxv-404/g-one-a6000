"""Gate E10, the two-regime perception rule.

From the build spec: "A still entity inside `R_NEAR` with clear line of sight is always reported; beyond
`R_NEAR` a still entity is reported only per the motion and probability rule; a moving entity is reported per
the same rule."

    near   = chebyshev(observer, cell) <= R_NEAR
    report = visible AND (near OR entity_moved_last_tick OR bernoulli(P_DISTANT_DETECT))

**Why this gate is load-bearing for the whole experiment.** The target behaviour is thanatosis, not crypsis.
Those are distinguishable only if stillness acts on the *capture decision* and never on *detection*. If a
still prey inside `R_NEAR` could go unreported, freezing would make it invisible, and a stillness result
would be concealment mislabelled as death-feigning. Every emergence claim would be about the wrong
phenomenon, and nothing downstream would reveal it.

So the near-regime assertion is the one that matters, and it is asserted positively (a still near prey IS
reported) and negatively (visibility is never reduced by stillness).
"""

import jax.numpy as jnp
import pytest

from configs.default import DEFAULT as P
from src.env.perception import build_vis_table, two_regime_report

R = P.r_pred
R_NEAR = P.r_near
NEVER = 1.0  # a draw of 1.0 is never below any probability in [0, 1]
ALWAYS = 0.0  # a draw of 0.0 is below any positive probability


@pytest.fixture(scope="module")
def open_table():
    """An obstacle-free table, so visibility is never the reason something is unreported."""
    return build_vis_table(jnp.zeros((16, 16), bool), R)


def report(table, observer, subject, moved, draw, p=P.p_distant_detect):
    return bool(
        two_regime_report(
            table,
            jnp.asarray(observer),
            jnp.asarray(subject),
            jnp.asarray(moved),
            jnp.asarray(draw),
            R,
            R_NEAR,
            p,
        )
    )


# ------------------------------------------------------- the near regime, the load-bearing half --


@pytest.mark.parametrize("offset", [(0, 1), (1, 1), (0, 2), (2, 2), (-2, 1), (-1, -2)])
def test_a_still_entity_inside_r_near_is_always_reported(open_table, offset):
    """The assertion the experiment's interpretation rests on.

    Every offset here is within `r_near` of 2 by Chebyshev distance. The entity has not moved and the
    probabilistic draw is set to never fire, so if the near rule were absent every one of these would be
    unreported and stillness would be concealment.
    """
    observer = (8, 8)
    subject = (8 + offset[0], 8 + offset[1])
    assert chebyshev_py(observer, subject) <= R_NEAR, "fixture error: this offset is not inside r_near"
    assert report(open_table, observer, subject, moved=False, draw=NEVER), (
        f"a still entity at offset {offset}, inside r_near with clear line of sight, was not reported. "
        "Stillness must never confer invisibility, or thanatosis becomes indistinguishable from crypsis"
    )


def test_stillness_never_reduces_reporting_anywhere(open_table):
    """The negative form: for every cell and every draw, moving cannot be reported LESS than still.

    A positive check that near-still is reported would pass a rule that also reported everything else. This
    asserts the direction of the effect across the whole patch, which is the polarity concern from
    CLAUDE.md section 11 applied to a perception rule.
    """
    observer = (8, 8)
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            subject = (8 + dr, 8 + dc)
            for draw in (ALWAYS, NEVER):
                still_seen = report(open_table, observer, subject, moved=False, draw=draw)
                moved_seen = report(open_table, observer, subject, moved=True, draw=draw)
                assert not (still_seen and not moved_seen), (
                    f"at offset ({dr},{dc}) with draw {draw} a still entity was reported while a moving one "
                    "was not. Motion must never reduce detection"
                )


# ------------------------------------------------------------------- the distant regime --


@pytest.mark.parametrize("offset", [(0, 3), (3, 3), (0, 4), (-4, 2)])
def test_a_still_distant_entity_is_reported_only_when_the_draw_fires(open_table, offset):
    observer = (8, 8)
    subject = (8 + offset[0], 8 + offset[1])
    assert chebyshev_py(observer, subject) > R_NEAR, "fixture error: this offset is inside r_near"

    assert not report(open_table, observer, subject, moved=False, draw=NEVER), (
        f"a still entity at offset {offset}, beyond r_near, was reported with the draw suppressed"
    )
    assert report(open_table, observer, subject, moved=False, draw=ALWAYS), (
        f"a still entity at offset {offset} was not reported even with the draw firing"
    )


@pytest.mark.parametrize("offset", [(0, 3), (3, 3), (0, 4)])
def test_a_moving_distant_entity_is_always_reported(open_table, offset):
    observer = (8, 8)
    subject = (8 + offset[0], 8 + offset[1])
    assert report(open_table, observer, subject, moved=True, draw=NEVER), (
        f"a moving entity at offset {offset} was not reported; motion alone must suffice"
    )


def test_the_probability_is_actually_consulted(open_table):
    """A construction check: the draw must change the answer, or `p_distant_detect` is inert.

    Before asserting on a rule, verify the rule responds to the input it claims to use. A version that
    ignored the draw entirely would pass several tests above.
    """
    observer, subject = (8, 8), (8, 4)
    seen = {report(open_table, observer, subject, moved=False, draw=d) for d in (ALWAYS, NEVER)}
    assert seen == {True, False}, (
        "the distant draw did not change the reporting decision, so p_distant_detect is inert"
    )


# --------------------------------------------------------------- visibility still dominates --


def test_an_occluded_entity_is_never_reported_however_near_or_fast():
    """Line of sight gates everything. The two-regime rule relaxes motion, never occlusion.

    Built with a wall so the subject is genuinely occluded, and the precondition is asserted rather than
    assumed, because a fixture whose wall does not actually block would make this test inert.
    """
    obstacles = jnp.zeros((16, 16), bool).at[8, 9].set(True)
    table = build_vis_table(obstacles, R)
    observer, subject = (8, 8), (8, 10)

    from src.env.perception import visible_from_table

    assert not bool(visible_from_table(table, jnp.asarray(observer), jnp.asarray(subject), R)), (
        "fixture precondition failed: the wall does not occlude, so this test cannot detect the bug"
    )
    for moved in (True, False):
        for draw in (ALWAYS, NEVER):
            assert not report(table, observer, subject, moved=moved, draw=draw), (
                f"an occluded entity was reported with moved={moved}, draw={draw}"
            )


def test_beyond_the_sensing_radius_nothing_is_reported(open_table):
    """`r_pred` bounds perception. The near rule must not leak past the table's edge."""
    observer = (8, 8)
    for subject in [(8, 8 + R + 1), (8 + R + 2, 8)]:
        for moved in (True, False):
            assert not report(open_table, observer, subject, moved=moved, draw=ALWAYS), (
                f"an entity at {subject}, beyond r_pred of {R}, was reported"
            )


def chebyshev_py(a, b):
    """Plain Python, so fixture preconditions do not depend on the code under test."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
