"""Gate E2 over step 6: the observation encoding and the network forward pass.

Two implementations of the same specification, compared on identical inputs. The observation is where a
`vmap` axis error or a patch-indexing mistake would produce a plausible but wrong tensor; the network is
where a GRU gate ordering or bias-structure mismatch would produce a network that trains perfectly well
while not being the specified one.

**The network half is the one that settles D-032.** The build specification states a parameter formula
assuming one bias vector per gate. A second implementation written from the equations disagrees immediately
if the structure differs, which turns the parameter count into a consequence of the comparison rather than a
separate claim nobody checked.

Independence is maintained the same way as elsewhere: the oracle recomputes visibility with its own
Fraction-based ray rule rather than reading the precomputed table, so a table built or indexed wrongly
disagrees rather than agreeing with itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT as P
from src.agent import network as fast_net
from src.env.observation import build_observation as fast_obs
from src.env.perception import build_vis_table
from src.reference import naive_env as oracle

H = W = 18
K = 4
N_TICKS = 40

CONFIGS = [
    dict(seed=0, density=0.10),
    dict(seed=1, density=0.20),
    dict(seed=2, density=0.05),
    dict(seed=3, density=0.30),
]


def _scenario(cfg):
    """One shared scene. Obstacles and positions are drawn once and handed to both implementations."""
    rng = np.random.default_rng(cfg["seed"])
    obstacles_n = rng.random((H, W)) < cfg["density"]
    free = np.argwhere(~obstacles_n)
    picks = rng.choice(len(free), size=K + 1, replace=False)

    return dict(
        obstacles_n=obstacles_n,
        obstacles_j=jnp.asarray(obstacles_n),
        vis_table=build_vis_table(jnp.asarray(obstacles_n), P.r),
        food=(rng.random((H, W)) < 0.05).astype(np.int32),
        target=(rng.random((H, W)) < 0.05).astype(np.int32),
        prey_pos=free[picks[:K]].astype(np.int32),
        pred_pos=free[picks[K]].astype(np.int32),
        prey_alive=rng.random(K) < 0.85,
        prey_moved=rng.random(K) < 0.5,
        prey_posture=rng.integers(0, 3, K).astype(np.int32),
        pred_moved=bool(rng.random() < 0.5),
        pred_appearance=rng.random(P.a_appearance).astype(np.float32),
        pools=rng.random((K, 3)).astype(np.float32) * np.array([P.e_max, P.s_max, P.h_max], np.float32),
    )


@pytest.mark.parametrize("cfg_i", range(len(CONFIGS)))
def test_observation_matches_the_oracle_for_every_agent(cfg_i):
    """Every agent's observation, element by element. Discrete content, so exact equality."""
    s = _scenario(CONFIGS[cfg_i])

    for k in range(K):
        o_j = fast_obs(
            obstacles=s["obstacles_j"],
            food_grid=jnp.asarray(s["food"]),
            target_grid=jnp.asarray(s["target"]),
            vis_table=s["vis_table"],
            self_pos=jnp.asarray(s["prey_pos"][k]),
            self_posture=jnp.int32(s["prey_posture"][k]),
            self_pools=jnp.asarray(s["pools"][k]),
            prey_pos=jnp.asarray(s["prey_pos"]),
            prey_alive=jnp.asarray(s["prey_alive"]),
            prey_moved=jnp.asarray(s["prey_moved"]),
            prey_posture=jnp.asarray(s["prey_posture"]),
            pred_pos=jnp.asarray(s["pred_pos"]),
            pred_moved=jnp.asarray(s["pred_moved"]),
            pred_appearance=jnp.asarray(s["pred_appearance"]),
            self_index=jnp.int32(k),
            params=P,
        )
        o_n = oracle.build_observation(
            obstacles=s["obstacles_n"],
            food_grid=s["food"],
            target_grid=s["target"],
            self_pos=s["prey_pos"][k],
            self_posture=s["prey_posture"][k],
            self_pools=s["pools"][k],
            prey_pos=s["prey_pos"],
            prey_alive=s["prey_alive"],
            prey_moved=s["prey_moved"],
            prey_posture=s["prey_posture"],
            pred_pos=s["pred_pos"],
            pred_moved=np.asarray(s["pred_moved"]),
            pred_appearance=s["pred_appearance"],
            self_index=k,
            params=P,
        )
        np.testing.assert_allclose(
            np.asarray(o_j),
            o_n,
            atol=P.e2_atol,
            rtol=P.e2_rtol,
            err_msg=f"observation for agent {k}, config {cfg_i}",
        )


def test_observation_coverage_is_sufficient():
    """The channels this gate claims to compare must actually be non-zero somewhere."""
    seen = np.zeros(P.c_channels, dtype=bool)
    for cfg in CONFIGS:
        s = _scenario(cfg)
        for k in range(K):
            o = oracle.build_observation(
                obstacles=s["obstacles_n"],
                food_grid=s["food"],
                target_grid=s["target"],
                self_pos=s["prey_pos"][k],
                self_posture=s["prey_posture"][k],
                self_pools=s["pools"][k],
                prey_pos=s["prey_pos"],
                prey_alive=s["prey_alive"],
                prey_moved=s["prey_moved"],
                prey_posture=s["prey_posture"],
                pred_pos=s["pred_pos"],
                pred_moved=np.asarray(s["pred_moved"]),
                pred_appearance=s["pred_appearance"],
                self_index=k,
                params=P,
            )
            patch = o[: P.s_patch * P.s_patch * P.c_channels].reshape(-1, P.c_channels)
            seen |= patch.max(axis=0) > 0.0

    missing = [i for i, ok in enumerate(seen) if not ok]
    assert not missing, f"channels {missing} were never non-zero across any scenario, so they are untested"


@pytest.mark.parametrize("cfg_i", range(len(CONFIGS)))
def test_network_forward_matches_the_oracle_over_a_sequence(cfg_i):
    """Recurrence compared tick by tick, so a divergence is reported where it starts.

    Run as a sequence rather than a single step, because a gate-ordering error can produce the correct
    output on the first tick from a zero hidden state and diverge only once the state is non-trivial.
    """
    w_j = fast_net.init_params(jax.random.key(cfg_i), P)

    # **Biases initialise to zero, so comparing the network as initialised leaves every bias path inert.**
    # A disagreement in how the two implementations apply a bias would then be invisible: adding a zero and
    # not adding it are the same number. Falsification found this, by a break that dropped a gate bias and
    # survived. Perturb every leaf, biases included, so the comparison exercises the path it claims to.
    pert = jax.random.split(jax.random.key(1000 + cfg_i), len(w_j))
    w_j = {
        k: v + 0.25 * jax.random.normal(pert[i], v.shape, v.dtype)
        for i, (k, v) in enumerate(sorted(w_j.items()))
    }
    for name in ("b_r", "b_z", "b_n", "b_in", "b_out"):
        assert float(jnp.max(jnp.abs(w_j[name]))) > 1e-3, (
            f"{name} is still effectively zero, so its path is not exercised by this comparison"
        )

    w_n = {k: np.asarray(v, dtype=np.float64) for k, v in w_j.items()}

    rng = np.random.default_rng(100 + cfg_i)
    h_j = fast_net.init_hidden(P)
    h_n = np.zeros(P.hd, dtype=np.float64)

    for t in range(N_TICKS):
        obs = rng.standard_normal(P.obs_dim).astype(np.float32)

        logits_j, h_j = fast_net.forward(w_j, jnp.asarray(obs), h_j)
        logits_n, h_n = oracle.forward(w_n, obs.astype(np.float64), h_n)

        np.testing.assert_allclose(
            np.asarray(logits_j),
            logits_n,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"logits diverged at tick {t}, config {cfg_i}",
        )
        np.testing.assert_allclose(
            np.asarray(h_j),
            h_n,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"hidden state diverged at tick {t}, config {cfg_i}",
        )
        assert int(fast_net.select_action(logits_j)) == oracle.select_action(logits_n), (
            f"the two implementations chose different actions at tick {t}"
        )


def test_the_hidden_state_actually_becomes_nontrivial():
    """Power check on the sequence test above.

    If the hidden state stayed near zero throughout, the comparison would only ever exercise the first-tick
    path and a recurrence error would survive. This asserts the sequence reaches a regime where the gates
    matter.
    """
    w = fast_net.init_params(jax.random.key(0), P)
    rng = np.random.default_rng(7)
    h = fast_net.init_hidden(P)
    for _ in range(N_TICKS):
        obs = jnp.asarray(rng.standard_normal(P.obs_dim).astype(np.float32))
        _, h = fast_net.forward(w, obs, h)
    assert float(jnp.max(jnp.abs(h))) > 0.05, (
        f"hidden state stayed near zero, max abs {float(jnp.max(jnp.abs(h))):.4f}, so the recurrent path "
        "is barely exercised"
    )


def test_the_parameter_count_is_a_consequence_not_a_claim():
    """D-032 made checkable. The formula, the arrays, and the oracle's view of the arrays must agree."""
    from configs.default import n_params as spec_n_params

    w = fast_net.init_params(jax.random.key(0), P)
    from_arrays = fast_net.n_params(w)
    from_formula = spec_n_params(P)
    from_oracle = int(sum(np.asarray(v).size for v in w.values()))

    assert from_arrays == from_formula == from_oracle, (
        f"arrays {from_arrays:,}, formula {from_formula:,}, oracle {from_oracle:,}. Per D-032 the "
        "specification changes, not the implementation."
    )
