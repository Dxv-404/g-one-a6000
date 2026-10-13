"""Build step 8's second half: all FIVE run configurations instantiate, and each can run an episode.

The criterion said four. `harmless_evolve` was added at step 12's specification and the arity assertion
moved with it, which is the gate failing for the right reason rather than the wrong one.

**Instantiating is not the criterion that matters.** A dataclass that constructs proves nothing about whether
the configuration describes a world the simulator can execute. `no_predator_evolve` is the case in point: it
constructed perfectly while `n_pred = 0` raised `IndexError` inside `run_episode`, which is DEV-013 and was
found by probing rather than by reading. So every configuration is RUN here, not just built.
"""

import jax
import jax.numpy as jnp
import pytest

from configs.run_configs import ALL, BY_NAME, NAIVE_EVAL, NO_PREDATOR_EVOLVE, SCRIPTED, check_invariants
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.entropy import PRNGSource
from src.env.perception import build_vis_table

# A shrunken world, as everywhere except E9. The configurations differ in `n_pred` and policy, not in scale,
# so shrinking preserves what is being tested.
SHRINK = dict(h=14, w=14, k=4, t_horizon=40, n_food_src=3, n_target_src=3)


def _small(cfg):
    from dataclasses import replace

    return replace(cfg.params, **SHRINK)


def _world(params, seed=0):
    ob = grid.make_obstacle_grid(jax.random.key(seed), params.h, params.w, params.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, params)
    return E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, params.r), food_src=fs, target_src=ts)


def test_the_declared_relationships_between_configurations_hold():
    """The invariants are in the module so they run outside the suite too, and are asserted here so they run
    in it. A relationship stated in prose is not checked; this one is."""
    check_invariants()


def test_there_are_exactly_five_and_they_are_named_as_specified():
    """**The arity moved from four to five, and the assertion moved with it. Owner's ruling, 2026-08-28.**

    Step 8's done criterion says "all four run configurations instantiate". Adding the third arm makes that
    five, and this gate failing on the change is the gate working: it is pinned as an EXACT set rather than a
    floor, so a configuration appearing or vanishing has to be a deliberate edit.

    `harmless_evolve` bounds the distribution-shift explanation that `naive_eval` cannot separate. Its own
    confound, that a harmless predator selects against responding, is recorded in the module docstring.
    """
    assert len(ALL) == 5
    assert set(BY_NAME) == {
        "scripted",
        "no_predator_evolve",
        "naive_eval",
        "random_policy",
        "harmless_evolve",
    }


@pytest.mark.parametrize("cfg", ALL, ids=[c.name for c in ALL])
def test_every_configuration_can_actually_run_an_episode(cfg):
    """**The criterion, and it is stronger than instantiation.**

    DEV-013: `no_predator_evolve` constructed fine and could not run, because the tick path indexed one
    predator and `n_pred = 0` gives an empty axis. Nothing in the suite caught it because nothing had ever
    run a value other than 1. A configuration that cannot execute is not a configuration.
    """
    params = _small(cfg)
    static = _world(params)
    weights = net.init_params(jax.random.key(1), params)
    state, acc = E.run_episode(jax.random.key(2), weights, static, PRNGSource(jax.random.key(3)), params)
    assert int(jnp.sum(acc.alive_ticks)) > 0, f"{cfg.name}: no agent was ever alive"
    assert state.prey_score.shape == (params.k,)


def test_the_predator_naive_world_really_contains_no_predator():
    """Asserted negatively, because absence is what this configuration IS.

    Every predator-conditioned accumulator must be exactly zero, and that is a stronger statement than
    checking `n_pred == 0`: it says the empty axis actually propagates as a no-op through every reduction
    rather than merely being declared.
    """
    params = _small(NO_PREDATOR_EVOLVE)
    static = _world(params)
    weights = net.init_params(jax.random.key(1), params)
    state, acc = E.run_episode(jax.random.key(2), weights, static, PRNGSource(jax.random.key(3)), params)
    assert state.pred_pos.shape[0] == 0, "the predator axis is not empty"
    for name in ("engaged_ticks", "visible_ticks", "near_ticks", "still_engaged", "accidental_still"):
        assert int(jnp.sum(getattr(acc, name))) == 0, (
            f"{name} is non-zero with no predator present, so the empty axis is not the no-op the "
            "predator-naive lineage depends on"
        )
    # And the descriptors that do NOT need a predator must still be live, or this arm measures nothing at all.
    assert float(jnp.sum(acc.density_sum)) > 0.0, (
        "grouping's accumulator is zero in the predator-free world, so the null would have no grouping "
        "baseline to compare against"
    )


def test_the_null_lives_in_the_treatments_world_not_its_own():
    """`naive_eval` changes the lineage and nothing else.

    If the worlds differed, a difference between evolved and null would not be attributable to predation
    selection, which is the only thing the comparison exists to isolate.
    """
    assert NAIVE_EVAL.params == SCRIPTED.params
    assert NAIVE_EVAL.params.n_pred == 1, "the null must be evaluated WITH a predator, or it is undefined"


def test_the_pair_declares_its_sequencing_dependency():
    """`naive_eval` cannot run before `no_predator_evolve` finishes, and that is declared rather than
    discovered. The build spec warns that discovering it at step twelve is the failure mode."""
    assert NAIVE_EVAL.weights_from == "no_predator_evolve"
    assert BY_NAME[NAIVE_EVAL.weights_from].evolves, (
        "naive_eval consumes weights from a configuration that does not evolve, so there would be no final "
        "weights to consume"
    )
    assert not NAIVE_EVAL.evolves


def test_a_configuration_cannot_both_evolve_and_inherit_weights():
    """The guard is in `__post_init__`, so it fires at construction rather than at run time."""
    from dataclasses import replace

    with pytest.raises(AssertionError, match="both evolves and inherits"):
        replace(NAIVE_EVAL, evolves=True)
