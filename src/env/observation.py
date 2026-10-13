"""The egocentric observation: a 9 by 9 patch of 14 channels, plus 6 scalars.

Build step 6. This is where `obs_dim` is fixed for the whole project at
`s_patch * s_patch * c_channels + N_SCALARS`, which is 1140. **Changing it after step 6 weights exist voids
every evolved weight**, exactly as `n_actions` does. The value is a derived property on `EnvParams` rather
than a literal here, so the network and this module cannot disagree about it.

Channels, from the build spec:

    0        obstacle
    1        food present
    2        forage target present
    3        predator present
    4        other prey present
    5        entity moved on the previous tick
    6        cell visible
    7,8,9    entity posture one-hot: upright, crouched, prone
    10..13   appearance feature vector, A = 4

Scalars: three normalised pools, then the observer's own posture as a three-way one-hot.

### Three things that look wrong and are not

**Channels 7 to 13 are populated and unused.** Posture is inert under the scripted predator, which does not
read it, and the appearance vector is not consumed by anything in increment 1. They are here because
increment 2 evolves a response to posture and increment 3 needs the appearance dimension, and because adding
channels later changes `obs_dim`. CLAUDE.md section 14.

**Out-of-bounds cells read as obstacles.** The grid boundary behaves like a wall, which matches
`grid.is_obstacle` and means a missing bounds check fails closed rather than open. Every other channel is
zero there.

**Visibility gates the entity channels but not the terrain channels.** An agent knows the shape of the walls
around it whether or not it can see through them, because obstacles are static and it has been there; it does
not know where an unseen prey is. Channel 6 carries the visibility mask itself so the network can tell "no
predator there" from "cannot see there", which are different states and would otherwise be identical zeros.

Reserved-channel note, because the analogy to reserved ACTIONS is tempting and wrong: a reserved output is a
free drift channel, because emitting it is selectively identical to emitting stay. A reserved input has no
such property. The network weights it or does not, and if it is uninformative selection simply does not
weight it.
"""

import jax
import jax.numpy as jnp
from jax import Array

# Channel indices, fixed by the build spec's observation table. Named because an off-by-one here produces a
# network that trains on the wrong feature and nothing downstream reports it.
CH_OBSTACLE = 0
CH_FOOD = 1
CH_TARGET = 2
CH_PREDATOR = 3
CH_PREY = 4
CH_MOVED = 5
CH_VISIBLE = 6
CH_POSTURE_FIRST = 7  # 7, 8, 9 upright, crouched, prone
CH_APPEARANCE_FIRST = 10  # 10 to 13 at a_appearance = 4

N_POSTURES = 3


def _patch_offsets(r: int) -> Array:
    """Every (dr, dc) in the patch, row-major, so the flatten order is stable and testable."""
    d = jnp.arange(-r, r + 1)
    dr, dc = jnp.meshgrid(d, d, indexing="ij")
    return jnp.stack([dr.reshape(-1), dc.reshape(-1)], axis=-1)


def build_observation(
    obstacles: Array,
    food_grid: Array,
    target_grid: Array,
    vis_table: Array,
    self_pos: Array,
    self_posture: Array,
    self_pools: Array,
    prey_pos: Array,
    prey_alive: Array,
    prey_moved: Array,
    prey_posture: Array,
    pred_pos: Array,
    pred_moved: Array,
    pred_appearance: Array,
    self_index: Array,
    params,
) -> Array:
    """One agent's observation, shape (obs_dim,). Single-agent; obtain the batch by `vmap`.

    `self_pools` is (energy, stamina, health) unnormalised; they are divided by their maxima here so the
    network sees values on a common scale regardless of the pool constants.

    Written single-environment and single-agent on purpose. The reference oracle is single-agent too, and
    hand-batching would make the differential harness awkward. CLAUDE.md section 11.
    """
    h, w = obstacles.shape
    r = params.r
    s = params.s_patch
    c = params.c_channels

    offsets = _patch_offsets(r)  # (S*S, 2)
    cells = self_pos[None, :] + offsets  # (S*S, 2)

    inside = (cells[:, 0] >= 0) & (cells[:, 0] < h) & (cells[:, 1] >= 0) & (cells[:, 1] < w)
    safe = jnp.stack([jnp.clip(cells[:, 0], 0, h - 1), jnp.clip(cells[:, 1], 0, w - 1)], axis=-1)

    patch = jnp.zeros((s * s, c), jnp.float32)

    # Terrain. Out of bounds reads as obstacle and as nothing else.
    obstacle = jnp.where(inside, obstacles[safe[:, 0], safe[:, 1]], True)
    patch = patch.at[:, CH_OBSTACLE].set(obstacle.astype(jnp.float32))
    patch = patch.at[:, CH_FOOD].set(
        jnp.where(inside, food_grid[safe[:, 0], safe[:, 1]] > 0, False).astype(jnp.float32)
    )
    patch = patch.at[:, CH_TARGET].set(
        jnp.where(inside, target_grid[safe[:, 0], safe[:, 1]] > 0, False).astype(jnp.float32)
    )

    # Visibility of each patch cell from the observer, from the precomputed table.
    from src.env.perception import visible_from_table

    vis = jax.vmap(visible_from_table, in_axes=(None, None, 0, None))(vis_table, self_pos, safe, r)
    vis = vis & inside
    patch = patch.at[:, CH_VISIBLE].set(vis.astype(jnp.float32))

    def _scatter(entity_pos, mask, channel, moved, posture, appearance):
        """Write one entity into whichever patch cell it occupies, if that cell is visible."""
        delta = entity_pos - self_pos
        in_patch = (jnp.max(jnp.abs(delta)) <= r) & mask
        idx = (delta[0] + r) * s + (delta[1] + r)
        idx = jnp.clip(idx, 0, s * s - 1)
        write = in_patch & vis[idx]
        f = write.astype(jnp.float32)

        out = patch.at[idx, channel].max(f)
        out = out.at[idx, CH_MOVED].max(f * moved.astype(jnp.float32))
        # Posture one-hot, and the appearance vector alongside it.
        for k in range(N_POSTURES):
            out = out.at[idx, CH_POSTURE_FIRST + k].max(f * (posture == k).astype(jnp.float32))
        for k in range(params.a_appearance):
            out = out.at[idx, CH_APPEARANCE_FIRST + k].max(f * appearance[k])
        return out

    # Other prey. The observer never writes itself into its own patch, which is what `self_index` excludes:
    # its own posture reaches the network through the scalars instead, and duplicating it in the patch would
    # give the same fact two encodings.
    zero_app = jnp.zeros((params.a_appearance,), jnp.float32)
    for k in range(prey_pos.shape[0]):
        patch = _scatter(
            prey_pos[k],
            prey_alive[k] & (k != self_index),
            CH_PREY,
            prey_moved[k],
            prey_posture[k],
            zero_app,
        )

    for k in range(pred_pos.shape[0] if pred_pos.ndim > 1 else 1):
        p = pred_pos[k] if pred_pos.ndim > 1 else pred_pos
        m = pred_moved[k] if pred_moved.ndim > 0 else pred_moved
        a = pred_appearance[k] if pred_appearance.ndim > 1 else pred_appearance
        patch = _scatter(p, jnp.asarray(True), CH_PREDATOR, m, jnp.int32(0), a)

    scalars = jnp.concatenate(
        [
            jnp.asarray(
                [
                    self_pools[0] / params.e_max,
                    self_pools[1] / params.s_max,
                    self_pools[2] / params.h_max,
                ],
                jnp.float32,
            ),
            (jnp.arange(N_POSTURES) == self_posture).astype(jnp.float32),
        ]
    )

    return jnp.concatenate([patch.reshape(-1), scalars])
