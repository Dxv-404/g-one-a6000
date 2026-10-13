"""The evolved policy: a small recurrent network, written in plain JAX rather than a library.

    flatten(S, S, C) -> Linear(Hd) -> tanh -> GRUCell(Hd) -> Linear(n_actions)

Build step 6. This is the last point at which `obs_dim` and `n_actions` are cheap to change; both are now
fixed, at 1140 and 22.

### Why this is hand-written when the build spec says Equinox or Flax

Neither is installed, and adding one means regenerating `uv.lock`. D-021 forbids that casually, because every
version is pinned to the environment the gates were verified in and re-resolving would move JAX, which voids
gate E1's determinism baseline. A GRU is twenty lines; a dependency change is a decision with a re-run of E1
attached.

The stated reason for preferring Equinox was that its PyTree-of-arrays model makes flattening weights for
evolution trivial. That property is not Equinox's: these weights are a plain dict of arrays, which
`jax.flatten_util.ravel_pytree` flattens for evosax at step 10 exactly as well. So the reason for the
dependency does not apply. See D-033.

**And it removes D-032's ambiguity by construction.** The build spec's parameter formula assumes one bias
vector per gate; Equinox, Flax and PyTree-style implementations disagree on that. Writing the cell here makes
the specification's convention the one in force rather than a coincidence. D-032 still applies in spirit: the
NumPy oracle checks this implementation against the specification, so a gate-ordering or bias mistake here is
caught rather than assumed away.

### Invariant I3

`init_hidden` returns exact zeros and is called at every episode start. Hidden state is never inherited and
never carried across episodes. The freshness half of I3 is testable now and is tested at step 6; the
inheritance half needs the evolutionary loop and is gate S8 at step 10. Two tests for one invariant, each
arriving when it becomes testable, rather than one arriving four steps after the code.
"""

import jax
import jax.numpy as jnp
from jax import Array

# GRU gate order, fixed here because it is exactly the sort of convention two implementations can disagree
# about while each stays internally consistent. Reset, update, candidate.
GATE_R, GATE_Z, GATE_N = 0, 1, 2


def init_params(key: Array, params) -> dict:
    """Initialise weights. Returns a plain PyTree of arrays, which `ravel_pytree` flattens for evosax.

    Glorot-uniform for every matrix and zeros for every bias, which is conventional and is not a scientific
    parameter: evolution replaces these immediately, and the initial distribution only sets generation zero.
    """
    k_in, k_r, k_z, k_n, k_hr, k_hz, k_hn, k_out = jax.random.split(key, 8)
    hd, n_act, obs = params.hd, params.n_actions, params.obs_dim

    def glorot(k, shape):
        limit = jnp.sqrt(6.0 / (shape[0] + shape[1]))
        return jax.random.uniform(k, shape, jnp.float32, -limit, limit)

    return {
        "w_in": glorot(k_in, (obs, hd)),
        "b_in": jnp.zeros((hd,), jnp.float32),
        # One bias vector per gate, matching the build spec's p_gru formula.
        "w_xr": glorot(k_r, (hd, hd)),
        "w_hr": glorot(k_hr, (hd, hd)),
        "b_r": jnp.zeros((hd,), jnp.float32),
        "w_xz": glorot(k_z, (hd, hd)),
        "w_hz": glorot(k_hz, (hd, hd)),
        "b_z": jnp.zeros((hd,), jnp.float32),
        "w_xn": glorot(k_n, (hd, hd)),
        "w_hn": glorot(k_hn, (hd, hd)),
        "b_n": jnp.zeros((hd,), jnp.float32),
        "w_out": glorot(k_out, (hd, n_act)),
        "b_out": jnp.zeros((n_act,), jnp.float32),
    }


def init_hidden(params) -> Array:
    """Exact zeros, every episode. **Invariant I3.**

    Not "small" and not "near zero". A freshness test asserts bitwise identity between two forward passes
    from separately initialised hidden states, and any nonzero initialisation would make that test either
    fail or need a tolerance, and a tolerance is exactly what would hide a carried remnant.
    """
    return jnp.zeros((params.hd,), jnp.float32)


def gru_cell(w: dict, x: Array, h: Array) -> Array:
    """One GRU step, in the standard formulation with one bias vector per gate.

        r = sigmoid(x @ w_xr + h @ w_hr + b_r)
        z = sigmoid(x @ w_xz + h @ w_hz + b_z)
        n = tanh(x @ w_xn + (r * h) @ w_hn + b_n)
        h' = (1 - z) * n + z * h

    **The reset gate multiplies the hidden state before the recurrent matmul, not after.** That is the one
    place implementations differ observably while both look correct, and it is why the oracle recomputes this
    independently rather than trusting the shape of the code.
    """
    r = jax.nn.sigmoid(x @ w["w_xr"] + h @ w["w_hr"] + w["b_r"])
    z = jax.nn.sigmoid(x @ w["w_xz"] + h @ w["w_hz"] + w["b_z"])
    n = jnp.tanh(x @ w["w_xn"] + (r * h) @ w["w_hn"] + w["b_n"])
    return (1.0 - z) * n + z * h


def forward(w: dict, obs: Array, h: Array) -> tuple[Array, Array]:
    """One tick. Returns (action_logits, new_hidden). Single-agent; obtain the batch by `vmap`."""
    x = jnp.tanh(obs @ w["w_in"] + w["b_in"])
    h_next = gru_cell(w, x, h)
    return h_next @ w["w_out"] + w["b_out"], h_next


def select_action(logits: Array) -> Array:
    """Deterministic argmax. **Ties break to the lowest index.**

    The build spec fixes argmax rather than sampling because every behavioural measure asks whether the agent
    *chose* an action, and under sampling any action may be noise rather than a decision. Under argmax,
    stillness is a deterministic function of the observed state.

    Ties will essentially never occur over float logits, but the oracle and this must break one the same way
    if it happens, or gate E2 fails on a coin flip and the failure is intermittent and baffling. `jnp.argmax`
    and `np.argmax` both return the lowest index; asserted in the tests rather than assumed.
    """
    return jnp.argmax(logits)


def n_params(w: dict) -> int:
    """Count the actual arrays, so the recorded budget describes the network that exists.

    `configs.default.n_params` computes the same figure from the specification's formula. They are compared
    in the tests, which is what makes D-032's pre-committed resolution actionable: if they disagree, the
    formula is wrong and the specification changes.
    """
    return int(sum(a.size for a in jax.tree_util.tree_leaves(w)))
