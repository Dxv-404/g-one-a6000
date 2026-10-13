"""The evolved policy and the hand-coded baselines it is measured against. Empty until build step 6.

Deliberately empty rather than absent. The directory documents the repository map in CLAUDE.md section 7, and
an empty namespace costs nothing while an undocumented gap invites someone to put the network somewhere else.

What arrives, and why the shape is already fixed:

    network   build step 6.  flatten(9,9,14) -> Linear(48) -> tanh -> GRUCell(48) -> Linear(n_actions).
              The parameter count is NOT repeated here. Run `uv run python configs/default.py`;
              n_params() is the single source and a transcribed number goes stale silently. DEV-010
              parameters. The architecture is a single disclosed prior and only weights evolve
    baseline  build step 9.  A hand-coded forager for gate S2 and a random policy for gate S9

`obs_dim` and `n_actions` are settled now rather than at step 6, because changing either later voids every
evolved weight and the parameter budget. That is why crouch, lie-down and the appearance vector already exist
and do nothing: they are shape, not features.

The hidden state is zeroed at every episode start and never inherited. Only weights cross generations, which
is invariant I3 and what separates evolved instinct from inherited learning.
"""
