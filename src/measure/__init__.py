"""Descriptors, learnability, and throughput. Empty until build step 11.

Deliberately empty rather than absent; see `src/agent/__init__.py` for why the markers exist.

Three constraints that are decided already and are easier to honour than to retrofit:

**Every confirmatory statistic must be an online accumulator.** `lax.scan` stacks its outputs in device
memory, so stacked per-tick observations would be 14.9 GB against a measured 6.17 GB usable, while the
accumulators are 0.46 MB. The accumulator design is what makes the scan fit at all rather than a performance
optimisation. At step 7, treat any scan output carrying a T axis as a defect until proven otherwise.

**Descriptors are differences, not raw rates.** Each is a threatened condition minus an unthreatened one,
which regresses out the agent's general tendency and makes strategies comparable. A raw stillness rate cannot
distinguish an agent that freezes at a predator from one that is simply inactive.

**Learnability is reported as undefined, never as zero, below `n_eng_min` engagements.** Zero would read as
collapse, which is a fail state in the ignition criterion, and the run would be abandoned for the wrong
reason. Low coverage means the world is not producing encounters, which is a calibration problem rather than
an evolutionary one.

Every measurement entry point here must call `src.assert_runtime_took_effect()`. The gates get it through
`tests/conftest.py`; nothing outside pytest gets it for free, and a run with a different memory ceiling would
record M2 against a ceiling no reported result ran under.

**Every recorded timing must state whether the compilation cache was cold or warm.** The cache lives outside
the repository, so a fresh clone legitimately starts cold and sees a slower figure. Measured on this machine:
the full suite is 890 seconds with no cache configured, 451 cold, 344 warm. Figures without that flag are not
comparable across machines. Record the cache size too, since the 2 GB cap silently stops applying if
`filelock` ever leaves the dependency set.

### The M4 sweep must iterate shape-changing parameters outermost. See D-026.

**JAX recompiles on shape change, not on value change**, and compilation is roughly 60 percent of runtime
here. Of the seven swept parameters only three change any array shape:

    H, W grid size          shapes    obstacle_grid, food_grid, target_grid, vis_table
    K prey per environment  shapes    every prey array
    R sensing radius        shapes    the observation patch
    obstacle_density        values    same shapes, different contents
    predator sprint uptime  values    float parameters
    p_distant_detect        values    float
    d_src                   values    placement only

So shape-changing parameters go in the outer loop and value-only parameters in the inner loop: one
compilation per distinct shape, reused across every value combination inside it. Ordered the other way the
sweep recompiles everything at every step, which is the difference between a sweep dominated by simulation
and one dominated by compilation.

It also removes the only realistic route to cache thrashing, since growth tracks distinct shapes rather than
sweep points. Free to honour if known in advance, awkward to retrofit once sweep code exists.
"""
