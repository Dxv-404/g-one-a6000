"""The environment: everything that is true of the world independent of any agent's policy.

Split by what changes and when, because that is what decides where a thing can live rather than by topic:

    grid        static for a run. Obstacle layout, bounds, solidity, movement legality
    perception  static for a run. Bidirectional visibility, precomputed into a table
    resources   per tick. Source layout, regulated regrowth, pick resolution
    dynamics    per tick. Action masking, the three body pools, write-once death
    entropy     the tagged source every stochastic decision draws through

The static-versus-per-tick split is load-bearing rather than tidy. Obstacles never change during a run, which
is what makes visibility a lookup rather than millions of ray-steps per tick, and source layout is closed over
at construction, which is why it consumes no entropy tag and is injected whole into the differential harness.

`predator` and `env` arrive at build steps 5 and 7. Step 5 is blocked pending a revised specification in which
cue-dependence becomes a swept axis rather than a fixed motion-gating rule.
"""
