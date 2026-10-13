"""The evolutionary loop and the fitness function.

    fitness   build step 10.  Invariant I1, made structural. Gate S1
    genome    build step 10.  Weight flattening. The structural half of invariant I3. Gate S8
    loop      NOT YET PRESENT. Blocked on a dependency decision, see below

**`loop` is absent and that is a blocker, not an oversight.** The build spec specifies evosax rather than a
hand-rolled genetic algorithm, and evosax is not in `pyproject.toml`, not in `uv.lock`, and not installed.
Adding it re-resolves a lockfile whose stated purpose is that `jax==0.11.0` cannot move, so it is a decision
for the owner rather than a step to take. See Q-035.

**`fitness` is the most constrained function in this repository and its signature is the constraint.** It
takes a single score array and no state object:

    def fitness(prey_score: Array) -> Array

Energy, stamina, health, elapsed steps and predator distance are therefore not in scope and cannot reach it.
That is invariant I1, and it is structural rather than a matter of discipline: a function that is not given
the state cannot read the state. An earlier draft passed the whole state and relied on a call-graph analysis
that three documents described and none implemented.

Widening that signature is the single most damaging change available here and it will look harmless when it
is proposed. Gate S1 exists as of build step 10 and does exactly that, in
`tests/scientific/test_s1_fitness_invariant.py`: it asserts the annotation, the parameter count, that
the module cannot reach environment state at all, and behaviourally that an agent surviving the full
horizon with zero collected targets scores exactly zero. Falsified 5 of 5.

`loop` is a genetic algorithm through evosax, tournament selection with elitism and Gaussian mutation and no
crossover. A GA rather than an evolution strategy because the primary result is a population-prevalence
statistic, which a population of distinct individuals expresses directly and a search distribution does not.
"""
