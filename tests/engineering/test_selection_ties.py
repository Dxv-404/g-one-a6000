"""Selection under ties: the elite cutoff and the tournament break ties uniformly at random. D-064.

**Why this file exists.** On a fitness plateau most comparisons are ties, so the tie rule decides which
genomes reproduce while every fitness-level statistic stays blind to it: a tie rule never changes the
expected fitness distribution, only which genotype inside a tied class goes forward. Run 30 measured the
elite cutoff tied in 106 of 120 generations, and `jax.lax.top_k` broke every one of those ties toward the
lowest position, where the sitting elites live. Nothing that looks at fitness could have seen it.

**Every uniformity test here is paired with the rule it replaced, and must FAIL that rule.** A uniformity
assertion that a biased selector also passes guards nothing, and the only proof that this one can fail is to
run it against the bias it exists to catch. The old rule is reproduced inside this file rather than imported,
so deleting the fix cannot also delete the comparison.

The statistical tests use fixed keys, so they are deterministic: a pass is reproducible and cannot flake.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from configs.default import DEFAULT
from src.evo import loop as L

P = DEFAULT.__class__(p_pop=64, n_elite=8, t_sel=3, sigma=0.01)
N_DRAWS = 4000

#: Strictness thresholds, classified in gate-thresholds.md. A fair selector must not be rejected at
#: P_FAIR; the replaced, position-biased selector must be rejected at P_BIASED. The two are eleven orders
#: of magnitude apart, so neither depends on where inside that gap the true p-value of a fair rule falls.
P_FAIR = 1e-3
P_BIASED = 1e-12


def _old_elite_indices(fit):
    """The rule D-064 replaced, reproduced here: `top_k` alone, ties to the lowest position."""
    return jax.lax.top_k(fit, P.n_elite)[1]


def _new_elite_indices(key, fit):
    return L.elite_indices(key, fit, P.n_elite)


def _counts(selected: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(selected.reshape(-1), minlength=n)


def _uniform_p(counts: np.ndarray) -> float:
    return float(stats.chisquare(counts).pvalue)


def _keys(n=N_DRAWS, seed=0):
    return jax.random.split(jax.random.key(seed), n)


# --------------------------------------------------------------------------------- elite cutoff --


def test_without_ties_the_elite_set_and_order_are_exactly_top_k():
    """Away from ties the fix must change nothing at all, not merely the set: same genomes, same order."""
    fit = jax.random.permutation(jax.random.key(1), jnp.arange(P.p_pop, dtype=jnp.float32))
    for key in _keys(50):
        np.testing.assert_array_equal(
            np.asarray(_new_elite_indices(key, fit)), np.asarray(_old_elite_indices(fit))
        )


def test_without_ties_a_whole_generation_is_bit_identical_to_the_old_rule():
    """The tie-break key is FOLDED from the generation key, not split from it, so the selection and mutation
    keys are unchanged. A generation with no tie at the cutoff must therefore be bit-identical."""
    genomes = jax.random.normal(jax.random.key(2), (P.p_pop, 5))
    fit = jax.random.permutation(jax.random.key(3), jnp.arange(P.p_pop, dtype=jnp.float32))
    key = jax.random.key(4)

    k_sel, k_mut = jax.random.split(key)
    old = jnp.concatenate(
        [
            genomes[_old_elite_indices(fit)],
            genomes[L.tournament(k_sel, fit, P.t_sel, P.p_pop - P.n_elite)]
            + P.sigma * jax.random.normal(k_mut, (P.p_pop - P.n_elite, 5)),
        ]
    )
    np.testing.assert_array_equal(np.asarray(L.next_population(key, genomes, fit, P)), np.asarray(old))


def test_all_tied_every_position_is_equally_likely_to_be_elite():
    """Every fitness equal: each position should hold an elite slot with probability n_elite / p_pop."""
    fit = jnp.zeros(P.p_pop)
    sel = np.asarray(jax.vmap(lambda k: _new_elite_indices(k, fit))(_keys()))
    counts = _counts(sel, P.p_pop)
    assert _uniform_p(counts) > P_FAIR, f"elite positions are not uniform under a full tie: {counts}"
    # The specific bias D-064 removed: the sitting elites' block, positions 0 .. n_elite - 1.
    block = counts[: P.n_elite].sum() / counts.sum()
    expected = P.n_elite / P.p_pop
    sd = np.sqrt(expected * (1 - expected) / counts.sum())
    assert abs(block - expected) < 5 * sd, (
        f"positions 0..{P.n_elite - 1} took {block:.4f} of elite slots against {expected:.4f} by chance"
    )


def test_partly_tied_the_strictly_fitter_always_win_and_the_tied_class_shares_the_rest_uniformly():
    """The interior case, between none tied and all tied, which is where run 30 lived: three genomes above
    the cutoff, twenty tied AT it (some inside the sitting-elite block), the rest below."""
    rng = np.random.default_rng(5)
    fit = np.zeros(P.p_pop, np.float32)
    above = np.array([10, 30, 50])
    tied = np.sort(
        np.concatenate([[0, 2, 5], rng.choice(np.setdiff1d(np.arange(8, P.p_pop), above), 17, False)])
    )
    fit[above] = 3.0
    fit[tied] = 1.0
    sel = np.asarray(jax.vmap(lambda k: _new_elite_indices(k, jnp.asarray(fit)))(_keys()))

    for row in sel[:200]:
        assert set(above) <= set(row.tolist()), "a strictly fitter genome lost its elite slot"
        assert set(row.tolist()) <= set(above) | set(tied.tolist()), "a genome below the cutoff became elite"
    tied_counts = _counts(sel, P.p_pop)[tied]
    assert _uniform_p(tied_counts) > P_FAIR, (
        f"the tied class did not share its slots uniformly: {tied_counts}"
    )


def test_the_uniformity_tests_reject_the_rule_they_replaced():
    """**The falsification, kept as a test.** The old rule is deterministic, so feeding it the same
    draws gives the same counts every time; the uniformity statistic must reject it emphatically on both the
    full tie and the partial tie. If this ever passes, the tests above have lost the power to see the bias."""
    full = np.tile(np.asarray(_old_elite_indices(jnp.zeros(P.p_pop))), (N_DRAWS, 1))
    assert _uniform_p(_counts(full, P.p_pop)) < P_BIASED

    fit = np.zeros(P.p_pop, np.float32)
    tied = np.arange(0, 40, 2)
    fit[tied] = 1.0
    part = np.tile(np.asarray(_old_elite_indices(jnp.asarray(fit))), (N_DRAWS, 1))
    assert _uniform_p(_counts(part, P.p_pop)[tied]) < P_BIASED


def test_on_a_plateau_the_same_elites_no_longer_persist_by_position():
    """The founder effect itself, across generations. Fitness is held flat; each genome carries its
    identity in every coordinate, and elites are copied unmutated, so the elite block's identities are
    readable each generation. Under the old rule the block never changes; under a fair rule a given elite
    keeps its slot only at chance, n_elite / p_pop."""
    genomes = jnp.repeat(jnp.arange(P.p_pop, dtype=jnp.float32)[:, None] * 1000.0, 3, axis=1)
    fit = jnp.zeros(P.p_pop)
    kept = []
    key = jax.random.key(9)
    prev = None
    for _ in range(60):
        key, k = jax.random.split(key)
        genomes = L.next_population(k, genomes, fit, P)
        block = set(np.round(np.asarray(genomes[: P.n_elite, 0]) / 1000.0).astype(int).tolist())
        if prev is not None:
            kept.append(len(block & prev) / P.n_elite)
        prev = block
    rate = float(np.mean(kept))
    # Chance is n_elite / p_pop = 0.125 for a sitting elite to be drawn again; mutation also breaks
    # identity for non-elites, which only lowers retention. The old rule gives exactly 1.0.
    assert rate < 0.4, f"sitting elites kept {rate:.2f} of their slots on a flat plateau; chance is 0.125"


def test_the_elite_tie_break_is_deterministic_for_a_key_and_varies_across_keys():
    fit = jnp.zeros(P.p_pop)
    a = np.asarray(_new_elite_indices(jax.random.key(11), fit))
    b = np.asarray(_new_elite_indices(jax.random.key(11), fit))
    c = np.asarray(_new_elite_indices(jax.random.key(12), fit))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "two keys gave the same tie-break on a full tie"


def test_under_ties_every_elite_is_at_least_as_fit_as_every_non_elite():
    rng = np.random.default_rng(13)
    for _ in range(20):
        fit = jnp.asarray(rng.integers(0, 4, P.p_pop).astype(np.float32))
        idx = np.asarray(_new_elite_indices(jax.random.key(int(rng.integers(1 << 30))), fit))
        rest = np.setdiff1d(np.arange(P.p_pop), idx)
        assert np.asarray(fit)[idx].min() >= np.asarray(fit)[rest].max()
        assert len(set(idx.tolist())) == P.n_elite, "an elite was selected twice"


# ----------------------------------------------------------------------------------- tournament --


def test_tournament_ties_are_uniform_over_the_population():
    fit = jnp.zeros(P.p_pop)
    winners = np.asarray(L.tournament(jax.random.key(14), fit, P.t_sel, 40000))
    assert _uniform_p(_counts(winners, P.p_pop)) > P_FAIR


def test_the_tournament_uniformity_test_rejects_a_lowest_index_tie_break():
    """Same falsification for the tournament: break ties toward the lowest GENOME index instead of the first
    drawn candidate, and the statistic must reject it."""
    cand = jax.random.randint(jax.random.key(14), (40000, P.t_sel), 0, P.p_pop)
    biased = np.asarray(jnp.min(cand, axis=1))
    assert _uniform_p(_counts(biased, P.p_pop)) < P_BIASED


def test_the_fraction_of_tournaments_decided_by_fitness_matches_its_identity():
    """**The tie-fraction identity.** With replacement and uniform draws, a size-t tournament is decided by
    fitness unless all t candidates share a fitness class: decided = 1 - sum_k p_k ** t. Asserted against the
    real candidate draws, so a non-uniform or without-replacement sampler would fail it."""
    rng = np.random.default_rng(15)
    fit = rng.choice([0.0, 0.0, 0.0, 0.0, 1.0, 2.0], P.p_pop).astype(np.float32)
    n = 40000
    cand = np.asarray(jax.random.randint(jax.random.key(16), (n, P.t_sel), 0, P.p_pop))
    f = fit[cand]
    decided = float(np.mean(~np.all(f == f[:, :1], axis=1)))
    shares = np.unique(fit, return_counts=True)[1] / P.p_pop
    expected = 1.0 - float(np.sum(shares**P.t_sel))
    sd = np.sqrt(expected * (1 - expected) / n)
    assert abs(decided - expected) < 5 * sd, f"decided {decided:.4f} against the identity's {expected:.4f}"

    # and the real tournament agrees with the draws it is built on: same key, same candidates
    winners = np.asarray(L.tournament(jax.random.key(16), jnp.asarray(fit), P.t_sel, n))
    np.testing.assert_array_equal(winners, cand[np.arange(n), np.argmax(f, axis=1)])
