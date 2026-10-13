"""Gates S2 and S9. Build step 9.

    S9  random-action agents score near zero and die frequently
    S2  a hand-coded forager beats random by a large margin

**These are tier two, and the distinction from tier one matters here.** An engineering gate asks whether the
code is correct. These ask whether the experiment can produce a trustworthy answer at all: if a competent
policy cannot outscore a random one in this world, then the environment cannot express or reward competence,
and every later null is uninterpretable. That is assumption A2 made testable.

**The forager is a measurement instrument and never touches the evolved lineage.** It is a hand-coded policy,
which CLAUDE.md section 1 refuses as a source of weights, imitation or initialisation. The protection is
structural: a policy consumes weights and returns actions, and nothing in `src/agent/baseline.py` returns
anything weight-shaped, so there is no direction in which it could seed a genome. Asserted below rather than
left as a claim.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT
from src.agent import baseline
from src.agent import network as net
from src.env import env as E
from src.env import grid, resources
from src.env.entropy import PRNGSource
from src.env.perception import build_vis_table

#: Full-size world, because these gates are about whether the REAL environment rewards competence. A shrunken
#: grid would change the travel distance between sources, which is exactly the quantity a forager exploits.
P = DEFAULT.__class__(t_horizon=400)

N_SEEDS = 8


def _world(seed: int):
    ob = grid.make_obstacle_grid(jax.random.key(seed), P.h, P.w, P.obstacle_density)
    fs, ts, _ = resources.place_sources(jax.random.key(seed), ob, P)
    return E.StaticEnv(obstacles=ob, vis_table=build_vis_table(ob, P.r), food_src=fs, target_src=ts)


def _forager_policy(weights, obs, hidden, step_index):
    """Adapter: the hand-coded forager over the prey batch. Everything else passes through untouched.

    That `weights` and `hidden` pass through UNUSED is the point. This policy has no parameters, so there is
    nothing for selection to act on and nothing an evolutionary loop could inherit from it.

    `step_index` is consumed and is not state: it drives the exploration heading, which is what makes the
    policy deterministic without being constant. See DEV-024.
    """
    del weights
    return jax.vmap(lambda o: baseline.forager(o, P, step_index))(obs), hidden


def _random_policy_for(key):
    """Adapter: uniform random actions, **stateless, keyed on the tick index**.

    **The first version of this kept a Python counter and was constant under `lax.scan`**, because the
    counter incremented once during tracing. Measured: one distinct action per prey across 400 ticks, and
    every tick identical to tick zero. Both S9 assertions passed anyway, since a constant policy also scores
    zero and also dies, so the gate was satisfied by precisely the opposite of what it claims to measure.

    Folding the tick index into the key makes the draw a function of the carry rather than of Python state,
    which is the same reason the environment threads `tick_index` through the scan instead of closing over it.
    """

    def policy(weights, obs, hidden, step_index):
        del weights
        k = jax.random.fold_in(key, step_index)
        acts = jax.vmap(lambda i: baseline.random_policy(jax.random.fold_in(k, i), P.n_actions))(
            jnp.arange(obs.shape[0])
        )
        return acts, hidden

    return policy


def _run(policy, seed: int):
    static = _world(seed)
    weights = net.init_params(jax.random.key(seed), P)
    state, acc = E.run_episode(
        jax.random.key(1000 + seed), weights, static, PRNGSource(jax.random.key(seed)), P, policy=policy
    )
    return {
        "score": float(jnp.sum(state.prey_score)),
        "alive_at_end": int(jnp.sum(state.prey_alive)),
        "mean_alive_ticks": float(jnp.mean(acc.alive_ticks)),
    }


@pytest.fixture(scope="module")
def results():
    rand = [_run(_random_policy_for(jax.random.key(s)), s) for s in range(N_SEEDS)]
    forg = [_run(_forager_policy, s) for s in range(N_SEEDS)]
    return rand, forg


# --------------------------------------------------- the check that would have caught the constant floor --
def test_the_random_floor_is_actually_random():
    """**S9's two assertions are both satisfied by a CONSTANT policy, so neither can detect one.**

    Score near zero: a constant policy scores zero. Dies frequently: a constant policy dies. The gate was
    green while the floor emitted one action per prey for all 400 ticks, because a Python tick counter inside
    the policy incremented at trace time and then never again under `lax.scan`.

    That is the same failure class as a differential agreeing over a path that never ran: the assertions were
    about the OUTCOME, and two very different instruments produce the same outcome. The only thing that
    separates them is measuring the instrument itself.

    So this checks the floor rather than its score, and it is deliberately placed BEFORE the two assertions
    that depend on it.
    """
    static = _world(0)
    weights = net.init_params(jax.random.key(0), P)
    _, acc, traj = E.run_episode(
        jax.random.key(1),
        weights,
        static,
        PRNGSource(jax.random.key(0)),
        P,
        record=True,
        policy=_random_policy_for(jax.random.key(0)),
    )
    emitted = np.asarray(traj.action_emitted)

    per_prey = [len(np.unique(emitted[:, i])) for i in range(P.k)]
    print(f"\n  S9 floor: distinct actions per prey over {P.t_horizon} ticks: {per_prey}")
    assert min(per_prey) > P.n_actions // 2, (
        f"the random floor emitted only {min(per_prey)} distinct actions for some prey across "
        f"{P.t_horizon} ticks. A uniform draw over {P.n_actions} actions should visit nearly all of them; "
        "this few means the policy is effectively constant and the floor is not a random floor."
    )

    identical = float(np.all(emitted == emitted[0][None, :], axis=1).mean())
    assert identical < 0.05, (
        f"{identical:.1%} of ticks were identical to the first tick. The policy is not varying with time, "
        "which is what a Python counter inside a scanned function produces."
    )

    moved = float(np.mean(acc.moved_ticks))
    assert moved > 0.1 * P.t_horizon, (
        f"random agents moved on {moved:.1f} of {P.t_horizon} ticks. A uniform policy emits a move action "
        "about 8 times in 22, so this low a rate means the actions are not uniform."
    )


# ------------------------------------------------------------------------------ S9, the floor --
def test_s9_random_agents_score_near_zero(results):
    """If a random policy scores well, the task is trivially solvable and carries no selection pressure.

    "Near zero" is asserted against the forager's score rather than against an absolute number, because an
    absolute threshold would be a value with no provenance, and this comparison is the thing the gate is
    actually about.
    """
    rand, forg = results
    r = float(np.mean([x["score"] for x in rand]))
    f = float(np.mean([x["score"] for x in forg]))
    print(f"\n  S9 random mean score {r:.2f}   forager mean score {f:.2f}")
    assert r < max(1.0, 0.25 * f), (
        f"random agents scored {r:.2f} against the forager's {f:.2f}. A task a random policy can do is a "
        "task with no selection gradient, and every later result would be measuring noise."
    )


def test_s9_random_agents_die_frequently(results):
    """The other half of the floor. Scoring nothing while surviving comfortably would mean the world has no
    survival pressure at all, which is a different failure from the task being trivial."""
    rand, _ = results
    survived = float(np.mean([x["alive_at_end"] for x in rand]))
    print(f"  S9 random survivors at horizon: {survived:.2f} of {P.k}")
    assert survived < P.k, (
        f"every prey survived the full horizon under random actions ({survived:.2f} of {P.k}). The world "
        "exerts no survival pressure, so nothing selects."
    )


# --------------------------------------------------------------- S2, the validating baseline --
def test_s2_the_forager_beats_random_by_a_large_margin(results):
    """**Assumption A2 made testable: can this environment express and reward competence at all?**

    If it cannot, no later comparison means anything, because a null that looks like the treatment would be
    indistinguishable from an environment in which nothing can be learned.
    """
    rand, forg = results
    r = float(np.mean([x["score"] for x in rand]))
    f = float(np.mean([x["score"] for x in forg]))
    assert f > 0, "the hand-coded forager scored nothing; the task may be unachievable as specified"
    assert f >= 5.0 * max(r, 0.5), (
        f"the forager scored {f:.2f} against random's {r:.2f}, which is not a large margin. Either the "
        "environment cannot reward competence or the baseline is not competent, and those need opposite "
        "responses: the first is a design failure and the second is a bug in the baseline."
    )


def test_s2_the_margin_is_consistent_across_seeds(results):
    """A margin that holds on average and fails on half the seeds is a margin driven by outliers."""
    rand, forg = results
    wins = sum(1 for a, b in zip(forg, rand, strict=True) if a["score"] > b["score"])
    print(f"  S2 forager beat random on {wins} of {N_SEEDS} seeds")
    assert wins >= N_SEEDS - 1, (
        f"the forager beat random on only {wins} of {N_SEEDS} seeds. An average margin carried by a few "
        "seeds is not evidence that the environment rewards competence."
    )


# --------------------------------------------------------------------- the structural refusal --
def test_the_baseline_module_cannot_produce_weights():
    """**CLAUDE.md section 1 refuses a hand-coded policy as a source of weights or imitation.**

    Held structurally rather than by discipline: every public callable here returns an action, and none
    returns anything weight-shaped. A function that returns an integer cannot seed a genome, which is the
    same argument that makes invariant I1 hold by a signature.
    """
    public = [n for n in dir(baseline) if not n.startswith("_")]
    assert "init_params" not in public, "the baseline module exposes a weight initialiser"

    obs = jnp.zeros((P.obs_dim,))
    a = baseline.forager(obs, P, jnp.int32(0))
    assert a.shape == (), "the forager returned something that is not a single action"
    assert jnp.issubdtype(a.dtype, jnp.integer), "the forager returned a non-integer, which could be weights"

    r = baseline.random_policy(jax.random.key(0), P.n_actions)
    assert r.shape == () and jnp.issubdtype(r.dtype, jnp.integer)


def test_the_forager_never_reads_the_predator_channel():
    """**Gate S2 measures competence at the TASK, not survival.** A baseline that also evaded the predator
    would demonstrate the thing the experiment exists to discover.

    Checked behaviourally rather than by reading the source: the action must be identical whether or not a
    predator is present in the patch.

    **The fixture leaves `CH_OBSTACLE` empty on purpose and that is load-bearing**, because it isolates the
    predator channel as the only thing varying. It also means the blocked-direction branch added in DEV-024
    cannot fire here, so this test says nothing about it; `tests/engineering/test_baseline.py` covers that.
    """
    s, c = P.s_patch, P.c_channels
    ctr = s // 2
    patch = jnp.zeros((s, s, c)).at[ctr, ctr + 2, 2].set(1.0)
    without = jnp.concatenate([patch.reshape(-1), jnp.zeros((P.N_SCALARS,))])
    with_pred = jnp.concatenate([patch.at[ctr, ctr - 1, 3].set(1.0).reshape(-1), jnp.zeros((P.N_SCALARS,))])
    assert int(baseline.forager(without, P, jnp.int32(0))) == int(
        baseline.forager(with_pred, P, jnp.int32(0))
    ), "the forager's action changed when a predator appeared, so it is not a pure foraging baseline"
