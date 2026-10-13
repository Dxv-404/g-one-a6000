"""Render one environment state to an RGB array, and optionally to a file.

The function returns the array rather than only writing a file, because the correctness test asserts against
the array. A renderer whose only output is an image can be checked by looking at it, and looking at it is
precisely the check that fails here: the picture looks right whether or not the cells are transposed.

### Layout, which the correctness test depends on

Each grid cell becomes a `CELL_PX` by `CELL_PX` block, with **row indexing row and column indexing column**,
so the array at `[r*CELL_PX : (r+1)*CELL_PX, c*CELL_PX : (c+1)*CELL_PX]` is the block for grid cell `(r, c)`.
That correspondence is the thing most worth asserting, because a transpose satisfies every other property a
render could plausibly be checked for: the same colours, the same counts, the same total ink.

Drawing order is fixed and later layers overwrite earlier ones:

    background -> obstacles -> source patches -> standing resources -> prey -> predator -> focus overlay

Predator over prey so a capture is visible rather than hidden, and resources over source patches so an item
is visible on the patch that produced it. The order matters for the test as well as for the eye: an assertion
about a cell has to know which layer won.

Matplotlib is used for the file output and the legend only. The array itself is built with NumPy, so the
correctness test does not depend on a rendering backend or on font availability.

Build increment 1.5, task 4.
"""

import numpy as np

# Cell size in pixels. An implementation constant: no measured result depends on it, and it is large enough
# that an off-by-one in the block arithmetic is visible rather than a single-pixel fringe.
CELL_PX = 12

# Colours as RGB floats in [0, 1]. Named rather than inline so the correctness test can assert against the
# same constants the renderer draws with, instead of hardcoding a colour and drifting from it.
CHANNEL = {
    "background": (0.97, 0.97, 0.95),
    "obstacle": (0.20, 0.20, 0.24),
    "food_source": (0.86, 0.93, 0.83),
    "target_source": (0.87, 0.88, 0.96),
    "food": (0.30, 0.65, 0.30),
    "target": (0.35, 0.40, 0.85),
    "prey": (0.95, 0.60, 0.15),
    "prey_dead": (0.75, 0.75, 0.72),
    "predator": (0.80, 0.15, 0.15),
    "visible": (1.00, 0.95, 0.70),
}


def _fill(img, r, c, colour):
    """Paint the block for grid cell (r, c). Row indexes row, column indexes column."""
    img[r * CELL_PX : (r + 1) * CELL_PX, c * CELL_PX : (c + 1) * CELL_PX] = colour


def _fill_inner(img, r, c, colour, inset=3):
    """Paint the middle of a cell, leaving the surrounding colour visible as a border.

    Used for entities and items so that what is underneath, a source patch or a visibility overlay, is not
    completely hidden by whatever stands on it.
    """
    img[
        r * CELL_PX + inset : (r + 1) * CELL_PX - inset,
        c * CELL_PX + inset : (c + 1) * CELL_PX - inset,
    ] = colour


def cell_block(img, r, c):
    """The pixel block belonging to grid cell (r, c). Exposed so tests assert through the same mapping.

    If the test computed the block itself it could agree with a transposed renderer by making the same
    mistake, which is the oracle-by-copying problem in miniature.
    """
    return img[r * CELL_PX : (r + 1) * CELL_PX, c * CELL_PX : (c + 1) * CELL_PX]


def render_state(
    obstacles,
    food_grid=None,
    target_grid=None,
    food_src=None,
    target_src=None,
    prey_pos=None,
    prey_alive=None,
    pred_pos=None,
    r_src=3,
    focus=None,
    vis_table=None,
    r=4,
):
    """Render one state to an (H*CELL_PX, W*CELL_PX, 3) float array in [0, 1].

    Everything except `obstacles` is optional, so this is usable from build step 1 onward rather than only
    once a full state exists. That matters because the point of the renderer is checking intermediate builds
    by eye, and a renderer that needs a complete state cannot be used until there is nothing left to check.

    `focus` is a prey index. When given with `vis_table`, the cells that prey can see are tinted, which is the
    overlay that makes an occlusion bug obvious: a wall with visible cells behind it is wrong in a way no
    numeric check was looking for.
    """
    obstacles = np.asarray(obstacles)
    h, w = obstacles.shape
    img = np.empty((h * CELL_PX, w * CELL_PX, 3), dtype=np.float32)
    img[:] = CHANNEL["background"]

    # Source patches first, so obstacles drawn over them read as solid rather than as tinted.
    for srcs, key in ((food_src, "food_source"), (target_src, "target_source")):
        if srcs is None:
            continue
        for s in np.asarray(srcs):
            r0, c0 = int(s[0]), int(s[1])
            for rr in range(max(0, r0 - r_src), min(h, r0 + r_src + 1)):
                for cc in range(max(0, c0 - r_src), min(w, c0 + r_src + 1)):
                    _fill(img, rr, cc, CHANNEL[key])

    for rr in range(h):
        for cc in range(w):
            if obstacles[rr, cc]:
                _fill(img, rr, cc, CHANNEL["obstacle"])

    # The focus agent's visible cells, before entities so entities stay legible on top.
    if focus is not None and vis_table is not None and prey_pos is not None:
        p = np.asarray(prey_pos)[focus]
        vt = np.asarray(vis_table)
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                tr, tc = int(p[0]) + dr, int(p[1]) + dc
                if 0 <= tr < h and 0 <= tc < w and vt[int(p[0]), int(p[1]), dr + r, dc + r]:
                    if not obstacles[tr, tc]:
                        _fill(img, tr, tc, CHANNEL["visible"])

    for grid, key in ((food_grid, "food"), (target_grid, "target")):
        if grid is None:
            continue
        g = np.asarray(grid)
        for rr in range(h):
            for cc in range(w):
                if g[rr, cc] > 0:
                    _fill_inner(img, rr, cc, CHANNEL[key])

    if prey_pos is not None:
        positions = np.asarray(prey_pos)
        alive = np.ones(positions.shape[0], dtype=bool) if prey_alive is None else np.asarray(prey_alive)
        for k in range(positions.shape[0]):
            key = "prey" if alive[k] else "prey_dead"
            _fill_inner(img, int(positions[k][0]), int(positions[k][1]), CHANNEL[key], inset=2)

    # Predator last so a capture shows the predator rather than hiding it under the prey it is on.
    if pred_pos is not None:
        for p in np.atleast_2d(np.asarray(pred_pos)):
            _fill_inner(img, int(p[0]), int(p[1]), CHANNEL["predator"], inset=2)

    return img


def save_state(path, title=None, **kwargs):
    """Render and write a PNG. Matplotlib is used here and nowhere else in this module.

    Kept separate from `render_state` so the correctness test never touches a rendering backend: a test that
    imported matplotlib would depend on font availability and a working display configuration for a property
    that is pure array arithmetic.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = render_state(**kwargs)
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=150)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return img
