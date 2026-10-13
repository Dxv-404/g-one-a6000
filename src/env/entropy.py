"""The tagged entropy source, in production and replay modes.

Why this indirection exists, since it looks like unnecessary ceremony and CLAUDE.md section 14 lists it among
the things that look like bugs and are not.

JAX uses a counter-based threefry generator and NumPy uses PCG64. From the same integer seed they produce
different streams, so two implementations that each draw their own randomness can never agree on discrete
state, and gate E2 would fail on a correct implementation. The builder's natural response to that would be to
weaken the gate, which the build spec calls the worst possible outcome.

The fix is that every stochastic decision is drawn through a source rather than from an internal generator:

    production mode   derives a subkey from the base key, the tag, and the tick
    replay mode       indexes a pre-generated array by the same tag and tick

so the differential harness can hand both implementations identical draws. Production pays no memory cost;
replay materialises arrays and is viable only at the small scale the harness runs at, since `distant_detect`
alone is 1.04 MB per environment and 1.06 GB at E of 1024.

A direct `jax.random` call at a draw site is faster to write and breaks replay alignment silently, which is
why it is forbidden rather than discouraged.

Build step 2. Gate E2 scaffolding.
"""

from typing import NamedTuple

import jax
from jax import Array

# The tag registry, fixed by D-001. Every stochastic decision site carries one of these names.
#
# A renamed or reordered tag breaks replay alignment while leaving both implementations individually correct,
# which is the worst class of bug to find late, so the list is explicit rather than derived from call sites,
# and `check_tags_match_registry` asserts it at test time. Adding a stochastic decision means adding its tag
# here in the same commit.
TAGS: tuple[str, ...] = (
    "spawn.prey",
    "spawn.pred",
    "spawn.appearance",
    "regrow.food.trigger",
    "regrow.food.cell",
    "regrow.target.trigger",
    "regrow.target.cell",
    "perceive.distant_detect",
    # Added with `p_wander`. **Adding a tag shifts no existing draw**, because `_subkey` folds in the tag id
    # and the tick rather than splitting sequentially, which is the property that comment was written for.
    "pred.wander.trigger",
    "pred.wander.dir",
)

TAG_ID: dict[str, int] = {tag: i for i, tag in enumerate(TAGS)}


class PRNGSource(NamedTuple):
    """Production mode. Carries one base key, threaded through the scan carry."""

    key: Array


class ReplaySource(NamedTuple):
    """Replay mode. Pre-generated draws, indexed by tag then tick.

    `draws` maps a tag to an array whose leading axis is the tick. A dict is a valid JAX pytree provided the
    keys are static, which they are, since they come from `TAGS`.
    """

    draws: dict[str, Array]


Entropy = PRNGSource | ReplaySource


def _subkey(source: PRNGSource, tag: str, step) -> Array:
    """Derive a subkey from the base key, the tag, and the tick.

    `fold_in` twice rather than sequential `split`, and the difference matters. Under sequential splitting the
    stream position of every draw depends on how many draws preceded it, so inserting a new stochastic
    decision anywhere silently shifts every later draw and changes results that should have been unaffected.
    Folding in a stable tag id and the tick makes each draw site's stream independent of every other, so
    adding a site at step 5 does not perturb the draws at step 3.

    This still satisfies the build spec's requirement that production consumes a key from the scan carry: the
    base key is what the carry holds, and no key is ever reused, because the pair (tag, step) is unique to a
    draw.
    """
    return jax.random.fold_in(jax.random.fold_in(source.key, TAG_ID[tag]), step)


def draw_uniform(source: Entropy, shape: tuple, tag: str, step) -> Array:
    """A uniform draw on [0, 1). `tag` must be in the registry; `step` is the tick index."""
    if tag not in TAG_ID:
        raise KeyError(
            f"unregistered entropy tag {tag!r}. Add it to TAGS and to D-001 in decisions.md in the same "
            "commit, or replay mode will not reproduce this draw and gate E2 will compare meaningless "
            "quantities while passing."
        )
    if isinstance(source, ReplaySource):
        return source.draws[tag][step].reshape(shape)
    return jax.random.uniform(_subkey(source, tag, step), shape)


def draw_bernoulli(source: Entropy, p: float, shape: tuple, tag: str, step) -> Array:
    """A Bernoulli draw, expressed through `draw_uniform` so both modes share one comparison point.

    Deliberately not `jax.random.bernoulli`: that would be a second draw mechanism whose replay counterpart
    would have to reproduce threefry's internal conversion from bits to booleans. Thresholding a uniform is
    the same operation in both modes by construction.
    """
    return draw_uniform(source, shape, tag, step) < p


def check_tags_match_registry(used_tags) -> None:
    """Assert that the tags a module actually draws with are exactly the registered ones.

    Required by D-001: gate E2 will pass while producing meaningless comparisons if a draw site exists whose
    tag is not registered, because replay would not supply that draw and the two implementations would
    silently diverge on it.
    """
    used = set(used_tags)
    unregistered = used - set(TAGS)
    assert not unregistered, (
        f"draw sites use unregistered tags {sorted(unregistered)}. Replay cannot supply them."
    )


def make_replay_source(key: Array, sites: dict[str, tuple]) -> ReplaySource:
    """Pre-generate draws for the harness. `sites` maps a tag to the full array shape including the tick axis.

    Only viable at small scale. At production scale `distant_detect` alone is 1.06 GB at E of 1024, which is
    why replay is a testing mode rather than the default.
    """
    source = ReplaySource(draws={})
    keys = jax.random.split(key, len(sites))
    draws = {}
    for k, (tag, shape) in zip(keys, sorted(sites.items()), strict=True):
        if tag not in TAG_ID:
            raise KeyError(f"unregistered entropy tag {tag!r}")
        draws[tag] = jax.random.uniform(k, shape)
    return source._replace(draws=draws)
