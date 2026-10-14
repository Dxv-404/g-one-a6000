"""The agent network: shape, parameter budget, invariant I3 freshness, and argmax determinism.

**The I3 freshness test is the one that matters here.** Hidden state must reset to exactly zero at every
episode start and never be inherited. Gate S8 checks the inheritance half behaviourally at step 10, which
leaves four build steps in which the invariant is live and unverified. The freshness half is testable now,
so it is tested now: two forward passes from separately zeroed states on the same input must be **bitwise
identical**, not close. A tolerance would hide exactly the leak the test exists to catch, because a small
carried remnant produces a small difference.

D-032's pre-committed resolution is made actionable by `test_the_specification_formula_matches_the_network`:
if the specification's parameter formula disagrees with the arrays that exist, the specification is wrong.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from configs.default import DEFAULT as P
from configs.default import n_params as spec_n_params
from src.agent import network as net


@pytest.fixture(scope="module")
def weights():
    return net.init_params(jax.random.key(0), P)


@pytest.fixture(scope="module")
def obs():
    return jax.random.normal(jax.random.key(1), (P.obs_dim,), jnp.float32)


# ------------------------------------------------------------------ shape and budget --


def test_obs_dim_is_what_the_config_derives():
    """1140 = 9 * 9 * 14 + 6, and the network reads it from the config rather than a literal."""
    assert P.obs_dim == P.s_patch * P.s_patch * P.c_channels + P.N_SCALARS
    assert P.obs_dim == 1140


def test_forward_produces_one_logit_per_action(weights, obs):
    h = net.init_hidden(P)
    logits, h_next = net.forward(weights, obs, h)
    assert logits.shape == (P.n_actions,) == (22,)
    assert h_next.shape == (P.hd,) == (48,)


def test_the_specification_formula_matches_the_network(weights):
    """D-032. If these disagree the specification changes, not the implementation.

    The formula assumes one bias vector per gate. This network is written that way deliberately, so the
    check should pass; it is here because "should pass" is exactly the claim that needs a test rather than
    an argument, and because a future change to either side must break something.
    """
    actual = net.n_params(weights)
    from_formula = spec_n_params(P)
    assert actual == from_formula, (
        f"the network has {actual:,} parameters and the specification formula gives {from_formula:,}. "
        "Per D-032 the specification changes, not the network: recompute n_params and let every consumer "
        "read it from there."
    )
    assert actual < 100_000, f"parameter budget breached: {actual:,}"


def test_the_gru_carries_one_bias_per_gate(weights):
    """The structural fact the formula depends on, asserted directly rather than inferred from a total.

    Two totals can agree by coincidence. This fails if a second bias vector appears, which is the specific
    divergence D-032 was written about.
    """
    biases = [k for k in weights if k.startswith("b_")]
    assert sorted(biases) == ["b_in", "b_n", "b_out", "b_r", "b_z"], (
        f"unexpected bias set {sorted(biases)}. Three gate biases, plus input and output, is the "
        "specification's structure"
    )
    for gate in ("b_r", "b_z", "b_n"):
        assert weights[gate].shape == (P.hd,)


# ---------------------------------------------------------------- invariant I3, freshness --


def test_a_fresh_hidden_state_is_exactly_zero():
    h = net.init_hidden(P)
    assert bool(jnp.all(h == 0.0)), "hidden state must initialise to exact zeros. Invariant I3"
    assert h.dtype == jnp.float32


def test_two_passes_from_separately_zeroed_states_are_bitwise_identical(weights, obs):
    """**Invariant I3, the half that is testable before the evolutionary loop exists.**

    Asserted as exact equality, not `allclose`. A carried remnant of a previous episode produces a small
    difference, and a tolerance is precisely what would hide it. See D-032.
    """
    logits_a, h_a = net.forward(weights, obs, net.init_hidden(P))
    logits_b, h_b = net.forward(weights, obs, net.init_hidden(P))

    assert bool(jnp.all(logits_a == logits_b)), (
        "two forward passes from separately zeroed hidden states produced different logits, so something "
        "carries across a reset. Invariant I3"
    )
    assert bool(jnp.all(h_a == h_b)), "the resulting hidden states differ, same reason"


def test_a_dirty_hidden_state_actually_changes_the_output(weights, obs):
    """The power check for the test above, and it is not optional.

    If the network ignored its hidden state, the freshness test would pass no matter what leaked. This
    asserts that hidden state is consulted at all, so that "identical from zeros" means something.
    """
    fresh, _ = net.forward(weights, obs, net.init_hidden(P))
    dirty, _ = net.forward(weights, obs, jnp.full((P.hd,), 0.3, jnp.float32))
    assert not bool(jnp.all(fresh == dirty)), (
        "the hidden state does not affect the output, so the freshness assertion is vacuous"
    )


def test_hidden_state_evolves_across_ticks(weights, obs):
    """Recurrence is real: feeding the same observation twice in sequence moves the state."""
    h0 = net.init_hidden(P)
    _, h1 = net.forward(weights, obs, h0)
    _, h2 = net.forward(weights, obs, h1)
    assert not bool(jnp.all(h1 == h2)), "the hidden state did not change on a second tick"
    assert not bool(jnp.all(h1 == h0))


# ------------------------------------------------------------------------- determinism --


def test_action_selection_is_deterministic_argmax(weights, obs):
    logits, _ = net.forward(weights, obs, net.init_hidden(P))
    a = net.select_action(logits)
    assert int(a) == int(jnp.argmax(logits))
    assert 0 <= int(a) < P.n_actions
    for _ in range(3):
        assert int(net.select_action(logits)) == int(a), "argmax is not deterministic"


def test_argmax_ties_break_to_the_lowest_index_in_both_libraries():
    """Asserted rather than assumed, because a disagreement here fails gate E2 intermittently.

    Ties will essentially never occur over float logits, but if one does, the oracle and the implementation
    must resolve it identically or the differential harness fails on a coin flip and the failure looks
    baffling rather than structural. One line, and it removes that possibility.
    """
    tied = jnp.array([1.0, 3.0, 3.0, 2.0, 3.0], jnp.float32)
    assert int(net.select_action(tied)) == 1, "JAX argmax did not break the tie to the lowest index"
    assert int(np.argmax(np.asarray(tied))) == 1, "NumPy argmax did not break the tie to the lowest index"
    assert int(net.select_action(tied)) == int(np.argmax(np.asarray(tied))), (
        "JAX and NumPy disagree on tie-breaking, so the oracle and the implementation would diverge"
    )


def test_weights_flatten_for_evosax(weights):
    """The property the build spec wanted Equinox for. A plain dict of arrays has it. See D-033."""
    from jax.flatten_util import ravel_pytree

    flat, unravel = ravel_pytree(weights)
    assert flat.ndim == 1
    assert flat.size == net.n_params(weights)
    restored = unravel(flat)
    for k in weights:
        assert bool(jnp.all(restored[k] == weights[k])), f"{k} did not survive a flatten and unflatten"


# ---------------------------------------------------------------------------------------------- Q-052, D-068

#: The batched logits' distance from a float64 reference, measured on the A6000 with this fixture: 8.7e-7 with
#: the network's matmuls pinned to HIGHEST, 1.1e-3 under JAX's default TF32. 1e-5 sits more than ten times
#: from each, so it separates the two states rather than tolerating either.
FULL_PRECISION_BOUND = 1e-5


def _precision_fixture():
    """Batched inputs with every term live: perturbed weights so no bias is zero, a non-zero hidden state."""
    k = jax.random.split(jax.random.key(7), 4)
    w = net.init_params(k[0], P)
    w = {
        name: v + 0.1 * jax.random.normal(jax.random.fold_in(k[1], i), v.shape)
        for i, (name, v) in enumerate(sorted(w.items()))
    }
    obs = jax.random.uniform(k[2], (256, P.obs_dim))
    h = 0.5 * jax.random.normal(k[3], (256, P.hd))
    return w, obs, h


def _float64_logits(w, obs, h):
    """The forward pass in NumPy float64, written from the GRU equations, not from the implementation."""
    w = {name: np.asarray(v, np.float64) for name, v in w.items()}
    obs, h = np.asarray(obs, np.float64), np.asarray(h, np.float64)

    def sig(v):
        return 1.0 / (1.0 + np.exp(-v))

    x = np.tanh(obs @ w["w_in"] + w["b_in"])
    r = sig(x @ w["w_xr"] + h @ w["w_hr"] + w["b_r"])
    z = sig(x @ w["w_xz"] + h @ w["w_hz"] + w["b_z"])
    n = np.tanh(x @ w["w_xn"] + (r * h) @ w["w_hn"] + w["b_n"])
    return ((1.0 - z) * n + z * h) @ w["w_out"] + w["b_out"]


def _batched_error(w, obs, h):
    logits, _ = jax.jit(jax.vmap(net.forward, in_axes=(None, 0, 0)))(w, obs, h)
    return float(np.max(np.abs(np.asarray(logits, np.float64) - _float64_logits(w, obs, h))))


def test_the_batched_forward_runs_at_full_float32_precision():
    """Q-052, ruled: the policy runs at full float32, not the GPU's TF32. Batched, as evolution evaluates it."""
    err = _batched_error(*_precision_fixture())
    assert err < FULL_PRECISION_BOUND, (
        f"batched logits are {err:.2e} from a float64 reference against a bound of {FULL_PRECISION_BOUND:.0e}. "
        "The network's matmuls are running at reduced precision, so evolution is selecting on "
        "hardware-rounded arithmetic. D-068."
    )


def test_no_precision_context_can_lower_the_network():
    """Pinned per call, so a caller's `default_matmul_precision`, or the matching environment variable, is inert."""
    # bfloat16 is the lowest setting JAX accepts here. "fastest" reads as if it should exist and does not.
    with jax.default_matmul_precision("bfloat16"):
        err = _batched_error(*_precision_fixture())
    assert err < FULL_PRECISION_BOUND, (
        f"under default_matmul_precision('bfloat16') the batched logits are {err:.2e} from float64. The pin "
        "is not per call, so something outside the network can change the arithmetic evolution runs on."
    )


def test_the_fixture_can_see_reduced_precision():
    """The two tests above are only evidence if this fixture shows TF32 when it is present. On the GPU it must.

    The same batched computation with plain `@`, at JAX's default precision. On a CPU there is no TF32 and
    nothing to see, so the precondition applies only on the GPU, which is where the run happens.
    """
    if jax.devices()[0].platform != "gpu":
        pytest.skip("no TF32 on a CPU, so the precondition has nothing to show")
    w, obs, h = _precision_fixture()

    def plain(obs_i, h_i):
        x = jnp.tanh(obs_i @ w["w_in"] + w["b_in"])
        r = jax.nn.sigmoid(x @ w["w_xr"] + h_i @ w["w_hr"] + w["b_r"])
        z = jax.nn.sigmoid(x @ w["w_xz"] + h_i @ w["w_hz"] + w["b_z"])
        n = jnp.tanh(x @ w["w_xn"] + (r * h_i) @ w["w_hn"] + w["b_n"])
        return ((1.0 - z) * n + z * h_i) @ w["w_out"] + w["b_out"]

    logits = jax.jit(jax.vmap(plain))(obs, h)
    err = float(np.max(np.abs(np.asarray(logits, np.float64) - _float64_logits(w, obs, h))))
    assert err > 10 * FULL_PRECISION_BOUND, (
        f"an unpinned batched forward is only {err:.2e} from float64 on this GPU, so this fixture cannot tell "
        "full precision from TF32 and the two tests above would pass either way."
    )
