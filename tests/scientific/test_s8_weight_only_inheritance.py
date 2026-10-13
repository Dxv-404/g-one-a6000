"""Gate S8. Weight-only inheritance, invariant I3. Build step 10.

The build spec's definition:

> Hidden state is exactly zero at every episode start; the inherited genome vector contains only weights and
> no activation-derived quantity; a fixed genome produces identical behaviour in episode one and episode two
> given the same environment seed.

**This is the gate most likely to be skipped and CLAUDE.md says so twice**, because a violation produces no
visible symptom. Nothing in a learning curve looks wrong when hidden state leaks across episodes. The run
simply gets better, which is what everybody wanted, and the result silently becomes a claim about inherited
learning rather than evolved instinct. That is the load-bearing claim of the whole clean-room frame.

**Three clauses, checked at the level each one actually lives at.**

    zero at start      init_hidden returns exact zeros, asserted on values not on a comment
    weights only       to_vector's signature admits nothing else, plus an exact round trip
    same behaviour     a fixed genome run twice at the same environment seed produces identical
                       trajectories, WITH A DIFFERENT EPISODE RUN IN BETWEEN

**The intervening episode is the point of the third clause and is easy to leave out.** Running the same
episode twice in a row would pass against a carry that happens to be reset by the identical inputs. Running
something different in between is what makes a leak observable: if any activation survives an episode
boundary, the second run of A is contaminated by X and diverges from the first.
"""

import jax
import jax.numpy as jnp
import numpy as np

from configs.default import DEFAULT
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.entropy import PRNGSource
from src.env.perception import build_vis_table
from src.evo import genome

P = DEFAULT.__class__(t_horizon=40)

COMPARED = ("prey_pos", "prey_alive", "prey_moved", "action_emitted", "action_executed", "pred_pos")


def _world(seed: int):
    ob = grid.make_obstacle_grid(jax.random.key(seed), P.h, P.w, P.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, P)
    return E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, P.r), food_src=fs, target_src=ts)


def _episode(weights, seed: int):
    """One episode at a fixed environment seed. Everything about the world comes from `seed`, so two calls
    with the same seed and the same weights must agree unless something persisted between them."""
    _, _, traj = E.run_episode(
        jax.random.key(1000 + seed), weights, _world(seed), PRNGSource(jax.random.key(seed)), P, record=True
    )
    return {name: np.asarray(getattr(traj, name)) for name in COMPARED}


# ------------------------------------------------------------------ 1. hidden state starts at zero --
def test_hidden_state_is_exactly_zero_at_episode_start():
    """Exactly zero, not nearly. A small non-zero initialisation would be a per-episode prior, and if it
    were ever derived from anything it would be inheritance through the back door."""
    h = net.init_hidden(P)
    assert np.array_equal(np.asarray(h), np.zeros_like(np.asarray(h))), (
        f"hidden state does not start at zero; max magnitude {float(jnp.max(jnp.abs(h)))}"
    )
    assert h.shape[-1] == P.hd, f"hidden state has width {h.shape[-1]}, expected hd = {P.hd}"


# ---------------------------------------------------------- 2. the genome carries weights and only weights --
def test_the_genome_vector_is_built_from_weights_and_nothing_else():
    """**Structural, in the same way invariant I1 is.** `to_vector` is not given hidden state, an episode,
    or a score, so no activation-derived quantity has a route in."""
    import inspect

    params = list(inspect.signature(genome.to_vector).parameters.values())
    assert len(params) == 1, (
        f"to_vector takes {len(params)} parameters: {[p.name for p in params]}. Invariant I3 is enforced by "
        "this signature; a second parameter is how an activation-derived quantity would enter the genome."
    )
    assert params[0].name == "weights"


def test_the_genome_round_trips_exactly():
    """A lossy genome would mean the offspring is not the parent's weights, which breaks inheritance in a
    way that looks like unusually strong mutation rather than like a bug."""
    w = net.init_params(jax.random.key(0), P)
    vec = genome.to_vector(w)
    back = genome.unflattener(w)(vec)

    assert set(back) == set(w), f"round trip changed the weight keys: {sorted(set(w) ^ set(back))}"
    for k in w:
        assert np.array_equal(np.asarray(w[k]), np.asarray(back[k])), f"{k} did not round trip exactly"
    assert genome.n_genes(w) == net.n_params(w), (
        f"the genome vector has {genome.n_genes(w)} entries but the network has {net.n_params(w)} "
        "parameters, so something is being carried that is not a weight, or a weight is being dropped"
    )


def test_the_template_supplies_structure_and_never_values():
    """`unflattener` takes a live weight tree as a template. This asserts none of that tree's numbers can
    reach the output, which is the plausible way a template becomes a leak."""
    a = net.init_params(jax.random.key(0), P)
    b = net.init_params(jax.random.key(1), P)
    rebuilt = genome.unflattener(a)(genome.to_vector(b))
    for k in b:
        assert np.array_equal(np.asarray(rebuilt[k]), np.asarray(b[k])), (
            f"{k} came back as the template's values rather than the vector's"
        )


# --------------------------------------------- 3. the behavioural clause, with a contaminating episode --
def test_a_fixed_genome_behaves_identically_in_episode_one_and_episode_two():
    """**The clause that catches a real leak, and the intervening episode is what gives it teeth.**

    Order is A, then X on a different seed and a different world, then A again. If any hidden state, PRNG
    position, or accumulator survives an episode boundary, the second A is contaminated by X and diverges.

    Running A twice in a row would not show it: an identical input sequence can mask a carry entirely.
    """
    w = net.init_params(jax.random.key(0), P)

    first = _episode(w, 0)
    contaminant = _episode(w, 7)
    second = _episode(w, 0)

    # Coverage, before the assertion that depends on it: if X behaved identically to A there was nothing to
    # contaminate with, and the intervening episode did no work.
    differing = sum(int(np.any(contaminant[name] != first[name])) for name in COMPARED)
    assert differing > 0, (
        "the intervening episode produced identical trajectories to the one it was meant to contaminate, so "
        "it cannot demonstrate isolation. Choose a seed that actually gives a different world."
    )

    for name in COMPARED:
        n = int(np.sum(first[name] != second[name]))
        assert n == 0, (
            f"{name} differs in {n} elements between the first and second run of the same genome at the "
            "same environment seed, with a different episode run in between. Something persisted across an "
            "episode boundary. This is invariant I3 broken, and it converts every result from a claim about "
            "evolved instinct into a claim about inherited learning."
        )


def test_a_different_genome_does_change_behaviour():
    """The complement. Without it, the gate above is satisfied by an environment that ignores the weights
    entirely, which would be green and meaningless in the way the taxonomy calls construction failure."""
    a = _episode(net.init_params(jax.random.key(0), P), 0)
    b = _episode(net.init_params(jax.random.key(1), P), 0)
    changed = sum(int(np.any(a[name] != b[name])) for name in COMPARED)
    assert changed > 0, (
        "two different genomes produced byte-identical trajectories in the same world, so behaviour does "
        "not depend on the weights and the isolation test above is vacuous"
    )
