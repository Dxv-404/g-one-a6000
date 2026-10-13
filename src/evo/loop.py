"""The genetic algorithm and the generation loop. Build step 10.

**Hand-rolled rather than through evosax, by an explicit owner decision.** The build spec named evosax; see
DEV-027 for the `amend-spec` deviation and D-039 for the reasoning. The short form: a library's defaults are
parameters nobody chose, and tournament size, mutation scale and elitism count would arrive as somebody
else's decisions inside a project whose established failure mode is exactly the unexamined constant. Every
value used here is in `configs/default.py` and passes the design audit.

The scheme, and it is small enough to read in one sitting:

    tournament selection at t_sel, with replacement
    elitism preserving the top n_elite UNCHANGED, unmutated
    Gaussian mutation at sigma on everything that is not an elite
    no crossover, matching the deep-neuroevolution convention

**Where the Python loop is allowed and where it is not.** The episode loop is `lax.scan`, always. The
GENERATION loop is a Python loop, deliberately: the build spec says a generation is one compiled call, so the
Python boundary sits between generations where it costs one dispatch per generation rather than one per tick.

**And the step is built once, outside that loop.** `make_generation_step` is a factory rather than a plain
function because a fresh callable per generation makes JAX re-lower and retain an executable every time. That
cost 31.5 MB and a fiftyfold slowdown per generation, and it killed a real run. See DEV-029.

**The DEV-023 hazard does not apply to that loop and it is worth saying why**, because the rule is stated
absolutely elsewhere. Python state frozen at trace time is a hazard for state read INSIDE a traced function.
The generation counter here is outside every `jit` boundary and is used to split keys before the call, so it
genuinely does advance. Anything inside `run_generation` takes its varying quantities from arguments.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields

import jax
import jax.numpy as jnp
from jax import Array

from src.env import entropy as ent
from src.env import env as E
from src.evo.fitness import fitness


def config_hash(params) -> str:
    """A stable fingerprint of the configuration. **Not Python's `hash`.**

    `hash()` on strings is salted per process unless `PYTHONHASHSEED` is set, so a config hash built from it
    would differ between runs of the same config and gate S7 would fail for a reason that has nothing to do
    with the science. This is the same class as any other environment-dependent value: it would look correct
    and be wrong somewhere else. sha256 over the sorted field list has no such dependence.

    Truncated to 16 hex characters, which is an implementation constant: it names runs in the registry and
    does not affect any result.
    """
    items = sorted((f.name, repr(getattr(params, f.name))) for f in fields(params))
    blob = "\n".join(f"{k}={v}" for k, v in items)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def tournament(key: Array, fit: Array, t_sel: int, n: int) -> Array:
    """`n` winners of independent size-`t_sel` tournaments, sampled with replacement.

    With replacement because that is the convention and because sampling without it would make the tournament
    size interact with the population size, which is a coupling nobody asked for. Ties go to the lower index
    via `argmax`, which is arbitrary but deterministic, and determinism is what E1 and S7 need.
    """
    candidates = jax.random.randint(key, (n, t_sel), 0, fit.shape[0])
    winners = jnp.argmax(fit[candidates], axis=1)
    return candidates[jnp.arange(n), winners]


def next_population(key: Array, genomes: Array, fit: Array, params) -> Array:
    """One generation of selection and mutation. Returns the new population, same shape.

    **Elites are copied unmutated**, which is what "preserving the top N_ELITE unchanged" means and is the
    property that makes best-fitness monotone across generations. A mutated elite is not an elite.
    """
    n_elite = params.n_elite
    n_offspring = params.p_pop - n_elite
    k_sel, k_mut = jax.random.split(key)

    elite_idx = jax.lax.top_k(fit, n_elite)[1]
    elites = genomes[elite_idx]

    parents = genomes[tournament(k_sel, fit, params.t_sel, n_offspring)]
    offspring = parents + params.sigma * jax.random.normal(k_mut, parents.shape)

    return jnp.concatenate([elites, offspring], axis=0)


def make_generation_step(unflatten, static: E.StaticEnv, params):
    """Build the generation step ONCE and call it per generation. **Not a convenience: the alternative leaks.**

    **DEV-029.** An earlier version defined the per-episode function inside `evaluate`, so every generation
    handed `vmap` a NEW function object. JAX keys its trace and lowering caches on callable identity, so each
    generation missed the cache, re-lowered, and retained another compiled executable. Measured on the same
    small configuration:

        fresh callable per generation    31.3 MB per generation, about 1.0 s per generation
        this, built once                 0.0 MB per generation, about 0.02 s per generation

    Zero growth and a fifty-fold speedup. The leaking version OOM-killed a 900-generation run at about
    generation 475, having reached 11.5 GB of host memory, which is 475 times 31.5 MB almost exactly.

    **The tell was visible and was read past**: per-generation time crept from 2.50 s to 2.76 s across those
    475 generations. A loop over fixed shapes should be flat.

    Returns `step(key, genomes) -> (next_genomes, fitness_of_the_population_passed_in)`.

    `E = p_pop * n_episodes` environments run as one batch. Gate E9 established the shape fits: 256 by 4 by 8
    prey by 400 ticks compiles in 13.1 s and reserves 0.25 GiB of a 6.00 GiB ceiling.

    **`params` and `static` are closed over rather than passed.** `EnvParams` is a frozen dataclass with no
    pytree registration and `vmap` rejects it even at `in_axes=None`. Closing over both is also what keeps the
    callable identity stable, which is the whole point of this factory.
    """

    def one_episode(vec: Array, k: Array) -> Array:
        # The reset source and the per-tick draws must be distinct keys: `run_episode` derives the first from
        # `key` and the second from `entropy`, and splitting here keeps that visible rather than relying on
        # the caller to remember.
        k_reset, k_draws = jax.random.split(k)
        state, _ = E.run_episode(k_reset, unflatten(vec), static, ent.PRNGSource(k_draws), params)
        return state.prey_score

    batched = jax.jit(jax.vmap(jax.vmap(one_episode, in_axes=(None, 0)), in_axes=(0, 0)))

    def step(key: Array, genomes: Array):
        k_eval, k_breed = jax.random.split(key)
        keys = jax.random.split(k_eval, params.p_pop * params.n_episodes)
        keys = keys.reshape(params.p_pop, params.n_episodes)
        fit = fitness(batched(genomes, keys))
        return next_population(k_breed, genomes, fit, params), fit

    return step


def initial_population(key: Array, template: dict, params) -> Array:
    """Generation zero: `p_pop` independently initialised networks, flattened.

    Independently rather than one network plus noise, so generation zero carries the diversity `init_params`
    provides rather than a cloud around a single point.
    """
    from src.agent import network as net
    from src.evo.genome import to_vector

    del template
    keys = jax.random.split(key, params.p_pop)
    return jax.vmap(lambda k: to_vector(net.init_params(k, params)))(keys)
