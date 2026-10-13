"""The NumPy oracle. Slow and duplicative on purpose, and load-bearing rather than decorative.

The dominant bug class in vectorised simulation is a vectorisation error that produces plausible but wrong
dynamics, and unit tests do not catch those reliably. The defence is an independent implementation of the
same specification, written to be read rather than to be fast, compared against the vectorised one across a
suite of seeds and configurations.

**Its slowness is the property that makes it useful.** A Python loop over cells is transparently correct in a
way a double `vmap` is not, and that asymmetry is the whole point. Do not optimise it, do not merge it into
`src/env/`, and never write it by copying from there: two implementations of one expression agree with each
other whether or not the expression is right.

That is not hypothetical. Writing the ray rule the obvious way, with `round` on a float, disagreed with the
specification in twelve configurations, because Python rounds half to even and the rule is half away from
zero. The exact-rational version caught it. Had the oracle reused the vectorised expression, both would have
been wrong together and gate E2 would have passed forever while testing nothing.

Gate E2's job is catching vectorisation bugs, not specification bugs. The specification itself is validated
elsewhere, by E4's symmetry property and by direct property tests. That division only holds while the two
implementations stay genuinely independent.
"""
