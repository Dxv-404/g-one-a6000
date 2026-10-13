"""The tagged entropy source, and the D-001 registry assertion.

D-001 requires the tag list to be asserted against the environment module at test time, because gate E2 will
pass while producing meaningless comparisons if a draw site exists whose tag is not registered: replay would
not supply that draw, so the two implementations would silently diverge on it.

No dynamics draw randomness yet, so these tests exercise the source itself. They are written now rather than
at build step 3 because the properties are easier to state in isolation than once several call sites depend
on them.
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from src.env.entropy import (  # noqa: E402
    TAGS,
    PRNGSource,
    check_tags_match_registry,
    draw_bernoulli,
    draw_uniform,
    make_replay_source,
)

# Transcribed from D-001 in decisions.md. If this list and TAGS disagree, one of them was edited without the
# other, which is the exact drift D-001 exists to prevent.
D001_TAGS = (
    "spawn.prey",
    "spawn.pred",
    "spawn.appearance",
    "regrow.food.trigger",
    "regrow.food.cell",
    "regrow.target.trigger",
    "regrow.target.cell",
    "perceive.distant_detect",
    "pred.wander.trigger",
    "pred.wander.dir",
)


def test_registry_matches_d001():
    assert tuple(TAGS) == D001_TAGS, (
        "the entropy tag registry has drifted from D-001. A renamed or reordered tag breaks replay "
        "alignment while leaving both implementations individually correct."
    )


def test_tags_are_unique():
    assert len(set(TAGS)) == len(TAGS), "a duplicated tag would make two draw sites share a stream"


def test_unregistered_tag_is_refused():
    """Failing loudly is the point. A silent fallback here is what makes E2 decorative."""
    source = PRNGSource(key=jax.random.key(0))
    with pytest.raises(KeyError, match="unregistered"):
        draw_uniform(source, (4,), "spawn.not_a_real_tag", 0)


def test_production_draws_are_reproducible():
    a = draw_uniform(PRNGSource(key=jax.random.key(7)), (16,), "spawn.prey", 3)
    b = draw_uniform(PRNGSource(key=jax.random.key(7)), (16,), "spawn.prey", 3)
    assert bool(jnp.all(a == b)), "the same base key, tag, and tick must give the same draw"


def test_different_tags_give_different_streams():
    key = jax.random.key(7)
    a = draw_uniform(PRNGSource(key=key), (32,), "spawn.prey", 0)
    b = draw_uniform(PRNGSource(key=key), (32,), "spawn.pred", 0)
    assert not bool(jnp.all(a == b)), "two draw sites at the same tick must not share a stream"


def test_different_ticks_give_different_streams():
    key = jax.random.key(7)
    a = draw_uniform(PRNGSource(key=key), (32,), "regrow.food.cell", 0)
    b = draw_uniform(PRNGSource(key=key), (32,), "regrow.food.cell", 1)
    assert not bool(jnp.all(a == b)), "a key must never be reused across ticks"


def test_a_draw_site_is_independent_of_the_others():
    """The property that sequential key splitting would not give, and the reason for folding in a tag id.

    Under sequential splitting, the stream position of every draw depends on how many draws preceded it, so
    inserting a new stochastic decision at build step 5 would silently shift every later draw and change
    results that should have been unaffected. Here each site's stream is a function of its own tag and tick
    only, so drawing from one site cannot perturb another.
    """
    key = jax.random.key(11)
    alone = draw_uniform(PRNGSource(key=key), (8,), "perceive.distant_detect", 5)
    # Draw from several other sites and ticks first; the target draw must be unchanged.
    for tag in ("spawn.prey", "spawn.pred", "regrow.food.trigger"):
        for step in range(4):
            draw_uniform(PRNGSource(key=key), (8,), tag, step)
    after = draw_uniform(PRNGSource(key=key), (8,), "perceive.distant_detect", 5)
    assert bool(jnp.all(alone == after))


def test_replay_reproduces_the_injected_draws():
    """Replay must return exactly what was pre-generated, since that is what makes E2 alignment possible."""
    t, n = 6, 8
    source = make_replay_source(jax.random.key(3), {"perceive.distant_detect": (t, n)})
    for step in range(t):
        got = draw_uniform(source, (n,), "perceive.distant_detect", step)
        assert bool(jnp.all(got == source.draws["perceive.distant_detect"][step]))


def test_bernoulli_is_a_threshold_on_the_uniform():
    """Both modes must share one comparison point, so bernoulli is defined through the uniform draw.

    Using `jax.random.bernoulli` instead would be a second draw mechanism whose replay counterpart would have
    to reproduce threefry's internal bits-to-boolean conversion.
    """
    source = PRNGSource(key=jax.random.key(5))
    p = 0.25
    u = draw_uniform(source, (256,), "perceive.distant_detect", 0)
    b = draw_bernoulli(source, p, (256,), "perceive.distant_detect", 0)
    assert bool(jnp.all(b == (u < p)))


def test_check_tags_match_registry_catches_an_unregistered_site():
    check_tags_match_registry(["spawn.prey", "regrow.food.cell"])
    with pytest.raises(AssertionError, match="unregistered"):
        check_tags_match_registry(["spawn.prey", "capture.coinflip"])
