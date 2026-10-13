"""The NumPy reference oracle. Slow, looped, and obviously correct.

This module is load-bearing rather than decorative. The dominant bug class in vectorised simulation is a
vectorisation error that produces plausible but wrong dynamics, and unit tests do not catch those reliably.
The defence is an independent implementation of the same specification, written to be read rather than to be
fast, and compared against the vectorised one over a suite of seeds and configurations.

**Do not optimise this and do not delete it.** Its slowness is what makes it a usable oracle: a Python loop
over cells is transparently correct in a way a double `vmap` is not, and that asymmetry is the whole point.
CLAUDE.md section 14 lists this among the things that look like bugs and are not.

### Written from the specification, not from the JAX code

The division of labour matters. **Gate E2's job is catching vectorisation bugs, not specification bugs.** The
ray formula itself is validated elsewhere: by E4's symmetry property, by the measured single-ray asymmetry
rate, and by direct property tests. What E2 catches is a `vmap` axis error, a broadcasting mistake, or a mask
applied to the wrong axis.

That only works if the two implementations are genuinely independent. So this file implements the rule as
stated in D-010, using exact rational arithmetic, rather than reusing the integer identity
`(2*i*|d| + n) // (2*n)` that the JAX version uses for speed. The two are algebraically equal, which is
exactly why writing the second one differently is worth doing: if the integer shortcut is wrong for some
input, this disagrees rather than agreeing with itself.

If you find yourself pasting from `src/env/`, you have collapsed the two implementations into one and gate E2
has become decorative.

Build step 2. Gate E2.
"""

import math
from fractions import Fraction

import numpy as np

# Eight-direction movement, clockwise from north, as (d_row, d_col). Transcribed from the action-set
# specification and D-009, which fixes this ordering. A mismatch here against `src/env/grid.py` would make
# gate E2 compare different action semantics while each implementation stayed internally correct.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # 0  N
    (-1, 1),  # 1  NE
    (0, 1),  # 2  E
    (1, 1),  # 3  SE
    (1, 0),  # 4  S
    (1, -1),  # 5  SW
    (0, -1),  # 6  W
    (-1, -1),  # 7  NW
)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _round_half_away_from_zero(q: Fraction) -> int:
    """Round a non-negative exact rational, halves going away from zero.

    Exact rationals rather than floating point, because a float would introduce its own rounding at exactly
    the half-integer crossings this rule exists to decide, and those crossings are where the two
    implementations would diverge if either were wrong.
    """
    assert q >= 0, "magnitudes only; the sign is applied by the caller"
    return math.floor(q + Fraction(1, 2))


def interior_cells(a, b) -> list[tuple[int, int]]:
    """The cells strictly between a and b, per the ray rule in D-010.

    With n = max(|dy|, |dx|), the cell at step i is

        offset_k = sign(d_k) * round_half_away_from_zero(i * |d_k| / n)

    Steps 1 to n-1 are interior. Both endpoints are excluded, which is why an obstacle standing at b does not
    hide b itself but does hide what lies beyond it. See Q-010, answered.
    """
    dy = int(b[0]) - int(a[0])
    dx = int(b[1]) - int(a[1])
    n = max(abs(dy), abs(dx))

    cells = []
    for i in range(1, n):  # empty when n is 0 or 1, which is correct: adjacent cells have no interior
        oy = _sign(dy) * _round_half_away_from_zero(Fraction(i * abs(dy), n))
        ox = _sign(dx) * _round_half_away_from_zero(Fraction(i * abs(dx), n))
        cells.append((int(a[0]) + oy, int(a[1]) + ox))
    return cells


def ray_clear(obstacles: np.ndarray, a, b) -> bool:
    """True when no obstacle lies strictly between a and b. Not symmetric in general, by design."""
    for r, c in interior_cells(a, b):
        if obstacles[r, c]:
            return False
    return True


def visible(obstacles: np.ndarray, a, b) -> bool:
    """Bidirectional line of sight, per the build spec: visible(a,b) = ray_clear(a,b) AND ray_clear(b,a)."""
    return ray_clear(obstacles, a, b) and ray_clear(obstacles, b, a)


def build_vis_table(obstacles: np.ndarray, r: int) -> np.ndarray:
    """Visibility from every cell to every cell in its patch, by explicit loop.

    Returns (H, W, S, S) bool with S = 2r + 1, indexed as table[row, col, dr + r, dc + r]. Patch cells outside
    the grid are False.

    This is the function gate E2 most needs, because its vectorised counterpart is a double `vmap` and an
    axis error there would produce a table that is wrong in a structured, plausible-looking way.
    """
    h, w = obstacles.shape
    s = 2 * r + 1
    table = np.zeros((h, w, s, s), dtype=bool)
    for row in range(h):
        for col in range(w):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    tr, tc = row + dr, col + dc
                    if 0 <= tr < h and 0 <= tc < w:
                        table[row, col, dr + r, dc + r] = visible(obstacles, (row, col), (tr, tc))
    return table


def in_bounds(obstacles: np.ndarray, pos) -> bool:
    h, w = obstacles.shape
    return 0 <= pos[0] < h and 0 <= pos[1] < w


def is_obstacle(obstacles: np.ndarray, pos) -> bool:
    """Out-of-bounds reads as solid, so the grid boundary behaves like a wall and a missing bounds check
    fails closed rather than open."""
    if not in_bounds(obstacles, pos):
        return True
    return bool(obstacles[pos[0], pos[1]])


def legal_move(obstacles: np.ndarray, pos, direction, sprint: bool) -> bool:
    """legal = in_bounds(proposed) AND NOT obstacle(proposed) AND path_clear(pos, proposed).

    A sprint moves two cells, so its intermediate cell is pos + direction and must itself be free. For a
    diagonal sprint the intermediate cell is itself diagonal, so a diagonal sprint whose intermediate diagonal
    cell is an obstacle is illegal even when both orthogonal neighbours are free. The sprint does not clip
    corners.
    """
    step = 2 if sprint else 1
    intermediate = (pos[0] + direction[0], pos[1] + direction[1])
    destination = (pos[0] + direction[0] * step, pos[1] + direction[1] * step)

    if not in_bounds(obstacles, destination) or is_obstacle(obstacles, destination):
        return False
    if sprint and (not in_bounds(obstacles, intermediate) or is_obstacle(obstacles, intermediate)):
        return False
    return True


def resolve_move(obstacles: np.ndarray, pos, direction, sprint: bool) -> tuple[int, int]:
    """Apply the move if legal, otherwise stay put. Blocked movement does not slide along the obstacle."""
    if not legal_move(obstacles, pos, direction, sprint):
        return (int(pos[0]), int(pos[1]))
    step = 2 if sprint else 1
    return (int(pos[0]) + direction[0] * step, int(pos[1]) + direction[1] * step)


def free_cell_count(obstacles: np.ndarray) -> int:
    return int((~obstacles).sum())


# --------------------------------------------------------------------------- resources
#
# Build step 3 in the oracle. This is the first subsystem with stochastic dynamics, so it is the first real
# exercise of replay mode: the tagged entropy source was built at step 2 but had no consumer, because step 2
# had no stochastic dynamics at all.
#
# The alignment risk is not the uniform values, which are injected and therefore identical by construction.
# It is the indexing convention. Both implementations must agree on which draw belongs to which decision, and
# on what "the n-th available cell" means. Those are exactly the disagreements that leave two implementations
# individually correct and jointly useless, which is the failure gate E2 exists to catch.


class ReplayReader:
    """The replay path of the entropy source, in NumPy, so the oracle stays JAX-free.

    Mirrors `draw_uniform` in replay mode: index the tag, then the tick, then reshape. Written to the same
    stated convention rather than by importing the JAX one, for the reason in this module's header.
    """

    def __init__(self, draws: dict):
        self.draws = draws

    def uniform(self, shape, tag: str, step: int) -> np.ndarray:
        if tag not in self.draws:
            raise KeyError(
                f"replay has no draws for tag {tag!r}. Replay would silently supply nothing and the two "
                "implementations would diverge on an undrawn decision."
            )
        return np.asarray(self.draws[tag][step]).reshape(shape)


def _patch_cells(centre, r_src: int, h: int, w: int) -> np.ndarray:
    """Flat indices within Chebyshev radius r_src of centre, in row-major order."""
    rows = np.arange(h).reshape(-1, 1)
    cols = np.arange(w).reshape(1, -1)
    inside = (np.abs(rows - int(centre[0])) <= r_src) & (np.abs(cols - int(centre[1])) <= r_src)
    return inside.reshape(-1)


def local_count(grid: np.ndarray, centre, r_src: int) -> int:
    """Standing items within r_src of a source. The quantity the cap regulates."""
    h, w = grid.shape
    return int(grid.reshape(-1)[_patch_cells(centre, r_src, h, w)].sum())


def regrow(
    replay: ReplayReader,
    grid: np.ndarray,
    sources: np.ndarray,
    obstacles: np.ndarray,
    cap: int,
    r_src: int,
    p_regrow: float,
    step: int,
    tag_trigger: str,
    tag_cell: str,
):
    """One tick of source-regulated regrowth. Returns (new_grid, placed_per_source).

    The rule as specified:

        for each source s:
            local = sum of grid within radius r_src of s
            if local < cap:
                with probability p_regrow per tick,
                place one item at a uniformly sampled free cell within r_src of s

    Sources are processed in order, because overlapping patches would otherwise both place into the same cell
    on the same tick and one placement would vanish.

    The cell is chosen by rank: one uniform selects the n-th currently-available cell, counted in row-major
    order over the flattened grid. Row-major because that is the order the flattened grid is written in, and
    because both implementations must agree on what "n-th" means or they diverge while each stays correct.
    """
    h, w = obstacles.shape
    free = (~obstacles).reshape(-1)
    grid_flat = grid.reshape(-1).copy()

    triggers = replay.uniform((sources.shape[0],), tag_trigger, step)
    cell_draws = replay.uniform((sources.shape[0],), tag_cell, step)

    placed = np.zeros(sources.shape[0], dtype=np.int32)

    for s in range(sources.shape[0]):
        patch = _patch_cells(sources[s], r_src, h, w)
        local = int(grid_flat[patch].sum())
        available = free & patch & (grid_flat == 0)
        n_available = int(available.sum())

        want = (local < cap) and (float(triggers[s]) < p_regrow) and (n_available > 0)
        if not want:
            continue

        rank = int(np.floor(float(cell_draws[s]) * n_available))
        rank = min(max(rank, 0), n_available - 1)
        # The rank-th available cell, counted in row-major order.
        target = int(np.flatnonzero(available)[rank])
        grid_flat[target] = 1
        placed[s] = 1

    return grid_flat.reshape(h, w), placed


def resolve_pick(
    food_grid: np.ndarray,
    target_grid: np.ndarray,
    prey_pos: np.ndarray,
    prey_alive: np.ndarray,
    picking: np.ndarray,
):
    """Resolve the pick action. Returns (food_grid, target_grid, gained_score, ate_food).

    A picked target increments the score immediately and picked food adds energy immediately. There is no
    inventory. Both resources on one cell are both taken, since they are separate grids.

    Contention resolves to the lowest prey index: two living prey can share a cell and both pick, and the item
    must be consumed once. Prey are walked in index order, which is what makes that deterministic.
    """
    h, w = food_grid.shape
    food_flat = food_grid.reshape(-1).copy()
    target_flat = target_grid.reshape(-1).copy()

    n = prey_pos.shape[0]
    gained = np.zeros(n, dtype=np.int32)
    ate = np.zeros(n, dtype=bool)

    for k in range(n):
        if not (bool(prey_alive[k]) and bool(picking[k])):
            continue
        cell = int(prey_pos[k][0]) * w + int(prey_pos[k][1])
        if target_flat[cell] > 0:
            target_flat[cell] = 0
            gained[k] = 1
        if food_flat[cell] > 0:
            food_flat[cell] = 0
            ate[k] = True

    return food_flat.reshape(h, w), target_flat.reshape(h, w), gained, ate


# --------------------------------------------------------------------------- the predator --
#
# Build step 5. Written from the build spec's pseudocode and the revised cue-dependence rule, deliberately
# NOT from `src/env/predator.py`. A reference implementation transcribed from the thing it checks is an
# equality test against itself, and gate E2's whole value is that the two were derived separately.
#
# Visibility here is recomputed per pair through this module's own `visible`, with its Fraction-based ray
# rule, rather than through a precomputed table. Same reason.

NO_TARGET = -1


def chebyshev(a, b) -> int:
    """Chebyshev distance. One diagonal step costs one, matching eight-connected movement."""
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def two_regime_report(
    obstacles: np.ndarray,
    observer,
    subject,
    subject_moved: bool,
    draw: float,
    r: int,
    r_near: int,
    p_distant_detect: float,
) -> bool:
    """near = chebyshev <= R_NEAR; report = visible AND (near OR moved OR bernoulli(P_DISTANT_DETECT)).

    An entity inside `r_near` with clear line of sight is reported whether or not it moved. That clause is
    what keeps stillness acting on capture rather than on detection, so that the target behaviour is
    thanatosis and not crypsis.
    """
    if chebyshev(observer, subject) > r:
        return False
    if not visible(obstacles, observer, subject):
        return False
    near = chebyshev(observer, subject) <= r_near
    return bool(near or subject_moved or (float(draw) < p_distant_detect))


def select_target(current: int, reported, distances) -> int:
    """Current target if still reported, else the nearest reported prey, else NO_TARGET.

    Ties go to the lowest index, matching `resolve_pick`'s contention rule.
    """
    current = int(current)
    if current != NO_TARGET and bool(reported[current]):
        return current

    best, best_d = NO_TARGET, None
    for k in range(len(reported)):
        if not bool(reported[k]):
            continue
        d = int(distances[k])
        if best_d is None or d < best_d:
            best, best_d = k, d
    return best


def step_predator(
    obstacles: np.ndarray,
    pred_pos,
    pred_stamina: float,
    target: int,
    still: float,
    engage: int,
    prey_pos: np.ndarray,
    prey_alive: np.ndarray,
    prey_moved: np.ndarray,
    distant_draws,
    wander_trigger,
    wander_dir,
    params,
):
    """One predator tick. Returns a dict matching the vectorised implementation's keys.

    Order follows the build spec exactly: perceive, select, move, then evaluate capture against the
    `prey_moved` passed in, which is step 4 of the same tick. Invariant I5.
    """
    n = prey_pos.shape[0]
    pred_pos = (int(pred_pos[0]), int(pred_pos[1]))

    reported = [
        two_regime_report(
            obstacles,
            pred_pos,
            prey_pos[k],
            bool(prey_moved[k]),
            distant_draws[k],
            params.r_pred,
            params.r_near,
            params.p_distant_detect,
        )
        and bool(prey_alive[k])
        for k in range(n)
    ]
    distances = [chebyshev(pred_pos, prey_pos[k]) for k in range(n)]

    prev_target = int(target)
    prev_distance = distances[prev_target] if prev_target != NO_TARGET else float("inf")

    target = select_target(target, reported, distances)
    has_target = target != NO_TARGET
    target_idx = target  # kept, because disengagement below resets `target` to -1
    target_pos = (int(prey_pos[target][0]), int(prey_pos[target][1])) if has_target else pred_pos

    sprinted = False
    new_pos = pred_pos
    if has_target:
        distance = chebyshev(pred_pos, target_pos)
        sprinted = bool(distance > 1 and float(pred_stamina) >= params.s_min_burst)
        step_dir = (
            _sign(target_pos[0] - pred_pos[0]),
            _sign(target_pos[1] - pred_pos[1]),
        )
        new_pos = resolve_move(obstacles, pred_pos, step_dir, sprinted)
        new_pos = (int(new_pos[0]), int(new_pos[1]))
    elif float(wander_trigger) < float(params.p_wander):
        # Written from the specification: a uniform random step when no prey is reported. Never sprints,
        # because the wander is search rather than pursuit and draws on the same stamina the chase needs.
        # At p_wander 0.0 this branch never runs and the predator holds position, which is what it did
        # before the parameter existed.
        idx = int(float(wander_dir) * len(DIRECTIONS))
        idx = min(max(idx, 0), len(DIRECTIONS) - 1)
        step_dir = DIRECTIONS[idx]
        new_pos = resolve_move(obstacles, pred_pos, step_dir, False)
        new_pos = (int(new_pos[0]), int(new_pos[1]))

    cue = float(params.cue_dependence)
    adjacent = bool(has_target and chebyshev(new_pos, target_pos) <= 1)
    moved = bool(prey_moved[target]) if has_target else False

    damage = 0.0
    if adjacent:
        damage = float(params.damage) * (1.0 - cue * (1.0 - (1.0 if moved else 0.0)))

    still = float(still)
    if adjacent:
        still = 0.0 if moved else still + cue

    # Engagement, Q-021's rule, written from the specification rather than from `src/env/predator.py`.
    # Accumulate distance CLOSED, hold at adjacency, reset only on the distance increasing or the target
    # changing. The old rule counted closing ticks and reset on arrival, which made it unsatisfiable against
    # a sprinting predator and an arrival event rather than a state.
    distance_after = chebyshev(new_pos, target_pos) if has_target else float("inf")
    engage = int(engage)
    if not (target == prev_target and target != NO_TARGET):
        engage = 0
    elif distance_after < prev_distance:
        engage = engage + int(prev_distance - distance_after)
    elif distance_after <= 1:
        engage = engage + 1
    elif distance_after == prev_distance:
        pass
    else:
        engage = 0

    if still > params.disengage_steps:
        target, still, engage = NO_TARGET, 0.0, 0

    damage_per_prey = np.zeros(n, dtype=np.float32)
    if has_target:
        damage_per_prey[target_idx] = damage

    # Stamina, deliberately spelled out rather than shared with the prey path, because this file exists to
    # disagree with the implementation and a shared helper cannot.
    if sprinted:
        new_stamina = float(pred_stamina) - params.s_burst_pred
    else:
        new_stamina = float(pred_stamina) + params.s_regen
    new_stamina = min(max(new_stamina, 0.0), params.s_max)

    return {
        "pred_pos": np.array(new_pos, dtype=np.int32),
        "target": int(target),
        "still": float(still),
        "engage": int(engage),
        "sprinted": bool(sprinted),
        "stamina": float(new_stamina),
        "adjacent": bool(adjacent),
        "damage_per_prey": damage_per_prey,
        "reported": np.array(reported, dtype=bool),
    }


# ------------------------------------------------------------------ step 4 dynamics --
#
# Build step 4, written from the build specification's update equations and NOT from `src/env/dynamics.py`.
# This is the oracle debt that step 5 did not create and did not close: E2 covered grid, visibility,
# resources and then the predator, while masks, pools and death causes had no counterpart at all.
#
# The equations, quoted so the source of each line is visible:
#
#     energy'   = clip(energy - E_BASE - E_SPRINT * is_sprint + E_FOOD * ate_food, 0, E_MAX)
#     stamina'  = clip(stamina - S_BURST * is_burst + S_REGEN * (1 - is_burst), 0, S_MAX)
#     health'   = clip(health - damage + H_REGEN - STARVE_RATE * (energy_after <= 0), 0, H_MAX)
#                 where `damage` is the graded magnitude from the capture rule, summed over
#                 predators. Amended at DEV-031: this line used to read `DAMAGE * damaged` with a
#                 boolean, which contradicted the capture rule and discarded the cue term.
#     newly_dead = alive AND (health_after <= 0)
#     cause      = where(energy_after <= 0, STARVATION, PREDATION)
#
# Loops and scalars throughout, one agent at a time, because being obviously correct is the entire job.

ACTION_MOVE_FIRST, ACTION_MOVE_LAST = 0, 7
ACTION_SPRINT_FIRST, ACTION_SPRINT_LAST = 8, 15
ACTION_STAY, ACTION_CROUCH, ACTION_LIE_DOWN, ACTION_PICK = 16, 17, 18, 19
ACTION_ATTACK, ACTION_DROP = 20, 21

CAUSE_ALIVE, CAUSE_STARVATION, CAUSE_PREDATION = 0, 1, 2
POSTURE_UPRIGHT, POSTURE_CROUCHED, POSTURE_PRONE = 0, 1, 2


def is_burst(a) -> bool:
    """The eight sprints, and only those, draw stamina."""
    return ACTION_SPRINT_FIRST <= int(a) <= ACTION_SPRINT_LAST


def is_basic_move(a) -> bool:
    """The eight single-cell moves. Never gated, never charged. Invariant I4."""
    return ACTION_MOVE_FIRST <= int(a) <= ACTION_MOVE_LAST


def is_complex(a) -> bool:
    """Blocked at zero energy: the sprints, and item manipulation, which here is `pick`. See Q-013."""
    return is_burst(a) or int(a) == ACTION_PICK


def is_reserved(a) -> bool:
    """`attack` and `drop`, unavailable for the whole of increment 1. D-030.

    Not in the build specification, which predates D-030 and describes a twenty-action set. Implemented from
    the decision entry instead, and noted here because a reader comparing this file against that document
    will otherwise find an unexplained difference.
    """
    return int(a) in (ACTION_ATTACK, ACTION_DROP)


def mask_actions(actions, energy, stamina, params):
    """Block illegal actions, falling through to stay. Returns (masked_actions, mask_events).

    Three rules: bursts blocked below `s_min_burst`, complex actions blocked at zero energy, reserved
    actions blocked always. A blocked action becomes stay and the event is recorded.

    **Basic movement is never masked here.** A move an obstacle refuses is still a move action whose
    position happens not to change, which is movement resolution rather than action masking. Invariant I4.
    """
    n = len(actions)
    masked = np.zeros(n, dtype=np.int32)
    blocked = np.zeros(n, dtype=bool)

    for k in range(n):
        a = int(actions[k])
        stop = (
            (is_burst(a) and float(stamina[k]) < params.s_min_burst)
            or (is_complex(a) and float(energy[k]) <= 0.0)
            or is_reserved(a)
        )
        blocked[k] = stop
        masked[k] = ACTION_STAY if stop else a

    return masked, blocked


def update_energy(energy, sprinting, ate_food, params):
    """Flat baseline drain every tick, an extra draw for sprinting, refilled by food, clipped into range.

    The baseline is charged whether or not the agent moved, so idling is never free. `e_sprint` is the one
    action-dependent term, which is why "flat regardless of action" is not an exact description of this.
    """
    out = np.zeros(len(energy), dtype=np.float32)
    for k in range(len(energy)):
        v = float(energy[k]) - params.e_base
        if bool(sprinting[k]):
            v -= params.e_sprint
        if bool(ate_food[k]):
            v += params.e_food
        out[k] = min(max(v, 0.0), params.e_max)
    return out


def update_stamina(stamina, bursting, params):
    """Drains on bursts only, regenerates otherwise. At zero it blocks bursts and never causes harm."""
    out = np.zeros(len(stamina), dtype=np.float32)
    for k in range(len(stamina)):
        v = float(stamina[k])
        v = v - params.s_burst_prey if bool(bursting[k]) else v + params.s_regen
        out[k] = min(max(v, 0.0), params.s_max)
    return out


def update_health(health, damage, energy_after, params):
    """Capture damage, slow passive recovery, and the starvation bleed.

    **`damage` is the MAGNITUDE, not a boolean. DEV-031.** The specification carried two equations that
    disagreed: a graded damage in the capture rule and a constant one here. This module implemented both
    faithfully, as did `src/env/`, so the differential agreed and the graded value never reached health.

    Written from the amended specification, not transcribed from `src/env/`.

    The bleed is keyed on energy AFTER its own update, per the specification ordering: starvation is
    evaluated against the energy the agent ends the tick with, not the energy it started with.
    """
    out = np.zeros(len(health), dtype=np.float32)
    for k in range(len(health)):
        v = float(health[k]) - float(damage[k]) + params.h_regen
        if float(energy_after[k]) <= 0.0:
            v -= params.starve_rate
        out[k] = min(max(v, 0.0), params.h_max)
    return out


def resolve_deaths(alive, health, energy, damaged, cause, death_step, starving_out, predated_out, step):
    """Write-once death, with two INDEPENDENT flags and the single cause derived from them.

    Written from the specification. The precedence rule alone records a starving agent that takes a lethal
    hit as starvation, which under-counts predation among the weakest agents. The two conditions are not
    alternatives, so both are recorded and `cause` is kept under the old precedence for convenience.

    Once set, nothing is revisited, which is what makes the record write-once rather than last-write-wins.
    An agent already dead is skipped entirely.
    """
    alive_out = np.array(alive, dtype=bool).copy()
    cause_out = np.array(cause, dtype=np.int32).copy()
    step_out = np.array(death_step, dtype=np.int32).copy()
    starve = np.array(starving_out, dtype=bool).copy()
    pred = np.array(predated_out, dtype=bool).copy()

    for k in range(len(alive)):
        if not bool(alive[k]):
            continue
        if float(health[k]) > 0.0:
            continue
        alive_out[k] = False
        is_starving = float(energy[k]) <= 0.0
        cause_out[k] = CAUSE_STARVATION if is_starving else CAUSE_PREDATION
        step_out[k] = int(step)
        if is_starving:
            starve[k] = True
        if bool(damaged[k]):
            pred[k] = True

    return alive_out, cause_out, step_out, starve, pred


def update_posture(posture, actions):
    """Crouch sets 1, lie-down sets 2, anything else returns the agent upright.

    **The build specification is silent on the third clause.** It says only "17 crouch, sets posture 1" and
    "18 lie down, sets posture 2", and does not say whether posture persists across other actions. This
    implements the non-persistent reading, matching `src/env/dynamics.py`, so that E2 tests the rest of the
    module rather than failing on an ambiguity. Raised as Q-020, because the two readings differ materially
    for increment 2: under persistence an agent could crouch once and remain crouched while moving.

    `posture` is accepted and ignored, which is the honest signature for a non-persistent rule: the argument
    exists so the call site does not change if the persistent reading is later chosen.
    """
    out = np.zeros(len(actions), dtype=np.int32)
    for k in range(len(actions)):
        a = int(actions[k])
        if a == ACTION_CROUCH:
            out[k] = POSTURE_CROUCHED
        elif a == ACTION_LIE_DOWN:
            out[k] = POSTURE_PRONE
        else:
            out[k] = POSTURE_UPRIGHT
    return out


# ------------------------------------------------------- step 6: observation and network --
#
# Build step 6 in the oracle. Written from the build specification's observation table and network
# definition, NOT transcribed from `src/env/observation.py` or `src/agent/network.py`.
#
# The network is mostly arithmetic, so the temptation is to skip the oracle on the grounds that a GRU is a
# GRU. That is exactly the case D-032 was written about: the specification states a parameter formula, an
# implementation does whatever it does, and nothing compares them. A second implementation written from the
# equations catches a gate-ordering mistake or a bias-structure mismatch immediately, and both are the kind
# of error that leaves a network that trains perfectly well while not being the specified one.

CH_OBSTACLE, CH_FOOD, CH_TARGET, CH_PREDATOR = 0, 1, 2, 3
CH_PREY, CH_MOVED, CH_VISIBLE = 4, 5, 6
CH_POSTURE_FIRST, CH_APPEARANCE_FIRST = 7, 10


def build_observation(
    obstacles,
    food_grid,
    target_grid,
    self_pos,
    self_posture,
    self_pools,
    prey_pos,
    prey_alive,
    prey_moved,
    prey_posture,
    pred_pos,
    pred_moved,
    pred_appearance,
    self_index,
    params,
):
    """One agent's observation as a flat vector, by explicit loop over patch cells.

    Visibility is recomputed here with this module's own `visible`, rather than read from a precomputed
    table, so the two implementations disagree if the table is built or indexed wrongly.
    """
    h, w = obstacles.shape
    r, s, c = params.r, params.s_patch, params.c_channels
    patch = np.zeros((s, s, c), dtype=np.float32)

    sr, sc = int(self_pos[0]), int(self_pos[1])

    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            tr, tc = sr + dr, sc + dc
            i, j = dr + r, dc + r

            if not (0 <= tr < h and 0 <= tc < w):
                # Out of bounds reads as solid and as nothing else, so a missing bounds check fails closed.
                patch[i, j, CH_OBSTACLE] = 1.0
                continue

            if obstacles[tr, tc]:
                patch[i, j, CH_OBSTACLE] = 1.0
            if food_grid[tr, tc] > 0:
                patch[i, j, CH_FOOD] = 1.0
            if target_grid[tr, tc] > 0:
                patch[i, j, CH_TARGET] = 1.0

            # Terrain above is reported regardless of sight; entities below are not.
            if visible(obstacles, (sr, sc), (tr, tc)):
                patch[i, j, CH_VISIBLE] = 1.0

    def place(pos, moved, posture, appearance, channel):
        pr, pc = int(pos[0]), int(pos[1])
        dr, dc = pr - sr, pc - sc
        if max(abs(dr), abs(dc)) > r:
            return
        i, j = dr + r, dc + r
        if patch[i, j, CH_VISIBLE] == 0.0:
            return
        patch[i, j, channel] = 1.0
        if moved:
            patch[i, j, CH_MOVED] = 1.0
        patch[i, j, CH_POSTURE_FIRST + int(posture)] = 1.0
        if appearance is not None:
            for k in range(params.a_appearance):
                patch[i, j, CH_APPEARANCE_FIRST + k] = max(
                    patch[i, j, CH_APPEARANCE_FIRST + k], float(appearance[k])
                )

    for k in range(prey_pos.shape[0]):
        if k == int(self_index) or not bool(prey_alive[k]):
            continue
        place(prey_pos[k], bool(prey_moved[k]), int(prey_posture[k]), None, CH_PREY)

    preds = pred_pos if np.asarray(pred_pos).ndim > 1 else np.asarray(pred_pos)[None, :]
    for k in range(preds.shape[0]):
        m = bool(np.ravel(pred_moved)[k]) if np.size(pred_moved) > 1 else bool(np.ravel(pred_moved)[0])
        app = pred_appearance if np.asarray(pred_appearance).ndim == 1 else pred_appearance[k]
        place(preds[k], m, 0, app, CH_PREDATOR)

    scalars = np.zeros(params.N_SCALARS, dtype=np.float32)
    scalars[0] = float(self_pools[0]) / params.e_max
    scalars[1] = float(self_pools[1]) / params.s_max
    scalars[2] = float(self_pools[2]) / params.h_max
    scalars[3 + int(self_posture)] = 1.0

    return np.concatenate([patch.reshape(-1), scalars]).astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_cell(w, x, h):
    """One GRU step from the specification's equations.

        r = sigmoid(x Wxr + h Whr + br)
        z = sigmoid(x Wxz + h Whz + bz)
        n = tanh(x Wxn + (r * h) Whn + bn)
        h' = (1 - z) n + z h

    **The reset gate multiplies the hidden state before the recurrent matmul.** That is the one detail
    implementations differ on while both look correct, which is why this is written out rather than reused.
    """
    r = _sigmoid(x @ w["w_xr"] + h @ w["w_hr"] + w["b_r"])
    z = _sigmoid(x @ w["w_xz"] + h @ w["w_hz"] + w["b_z"])
    n = np.tanh(x @ w["w_xn"] + (r * h) @ w["w_hn"] + w["b_n"])
    return (1.0 - z) * n + z * h


def forward(w, obs, h):
    """One tick. Returns (action_logits, new_hidden)."""
    x = np.tanh(obs @ w["w_in"] + w["b_in"])
    h_next = gru_cell(w, x, h)
    return h_next @ w["w_out"] + w["b_out"], h_next


def select_action(logits):
    """Deterministic argmax, ties to the lowest index. `np.argmax` does this, and the test asserts it."""
    return int(np.argmax(logits))


# --------------------------------------------------------------------- the episode loop --
#
# Build step 7 in the oracle. Written from the build specification's twelve-step tick order, composing the
# oracle functions already here rather than importing anything from `src/env/`.
#
# The value of an independent episode loop is not that each piece is re-derived, since the pieces already
# have their own differential coverage. It is that the ORDER is re-derived. A tick order that resolves
# capture before movement, or reads a stale `prey_moved`, produces a simulation that runs perfectly and
# answers a different question, and invariant I5 is exactly that ordering.


def tick(
    state: dict,
    acc: dict,
    weights: dict,
    obstacles: np.ndarray,
    food_src: np.ndarray,
    target_src: np.ndarray,
    entropy,
    step_index: int,
    params,
):
    """One tick in the specification's order. `state` and `acc` are plain dicts of NumPy arrays.

    The numbered comments are the specification's step numbers, kept so that a reordering is visible as a
    renumbering rather than as a diff nobody reads.
    """
    k = params.k
    n_actions = params.n_actions

    # 1. observations, one agent at a time.
    draws = entropy.draws["perceive.distant_detect"][step_index]
    wander_trigger_draws = entropy.draws["pred.wander.trigger"][step_index]
    wander_dir_draws = entropy.draws["pred.wander.dir"][step_index]
    obs = np.stack(
        [
            build_observation(
                obstacles=obstacles,
                food_grid=state["food_grid"],
                target_grid=state["target_grid"],
                self_pos=state["prey_pos"][i],
                self_posture=state["prey_posture"][i],
                self_pools=np.array(
                    [state["prey_energy"][i], state["prey_stamina"][i], state["prey_health"][i]]
                ),
                prey_pos=state["prey_pos"],
                prey_alive=state["prey_alive"],
                prey_moved=state["prey_moved"],
                prey_posture=state["prey_posture"],
                pred_pos=state["pred_pos"],
                pred_moved=np.array([False]),
                pred_appearance=state["pred_appearance"],
                self_index=i,
                params=params,
            )
            for i in range(k)
        ]
    )

    # 2. forward pass and argmax. This is action_emitted.
    logits = np.zeros((k, n_actions))
    hidden = np.zeros_like(state["prey_hidden"])
    for i in range(k):
        logits[i], hidden[i] = forward(weights, obs[i], state["prey_hidden"][i])
    action_emitted = np.array([select_action(logits[i]) for i in range(k)], dtype=np.int32)

    # 3. mask. action_executed is what the environment allowed.
    action_executed, _ = mask_actions(action_emitted, state["prey_energy"], state["prey_stamina"], params)

    # 4. movement, and prey_moved from whether the cell actually changed.
    prey_pos = state["prey_pos"].copy()
    sprinting = np.array([is_burst(a) for a in action_executed], dtype=bool)
    for i in range(k):
        if not bool(state["prey_alive"][i]):
            continue
        a = int(action_executed[i])
        if a <= 15:
            d = DIRECTIONS[a % 8]
            prey_pos[i] = resolve_move(obstacles, tuple(state["prey_pos"][i]), d, bool(sprinting[i]))
    prey_moved = np.any(prey_pos != state["prey_pos"], axis=-1) & state["prey_alive"]

    # 5. pick.
    picking = (action_executed == ACTION_PICK) & state["prey_alive"]
    food_grid, target_grid, gained, ate = resolve_pick(
        state["food_grid"], state["target_grid"], prey_pos, state["prey_alive"], picking
    )
    prey_score = state["prey_score"] + gained

    # 6 and 7. predator, with capture read against prey_moved from step 4 of THIS tick. Invariant I5.
    # **One call per predator, and the list is empty at `n_pred = 0`.** The oracle was single-predator until
    # step 8, so E2 could not witness `n_pred = 0`, which `no_predator_evolve` requires, nor `n_pred > 1`,
    # which the proposed sweep axis requires. Both DEV-016 fixes live in that same code path and had no
    # independent implementation to disagree with them, which is the shape DEV-014 had.
    preds = [
        step_predator(
            obstacles,
            state["pred_pos"][j],
            float(state["pred_stamina"][j]),
            int(state["pred_target"][j]),
            float(state["pred_still"][j]),
            int(state["pred_engage"][j]),
            prey_pos,
            state["prey_alive"],
            prey_moved,
            draws,
            wander_trigger_draws[j],
            wander_dir_draws[j],
            params,
        )
        for j in range(len(state["pred_pos"]))
    ]

    # Reductions over the predator axis, written as explicit loops that are simply skipped when the list is
    # empty. A prey is damaged if ANY predator damages it and reported if ANY predator reports it.
    n_prey = len(prey_pos)
    damage_total = np.zeros(n_prey, dtype=np.float32)
    reported_any = np.zeros(n_prey, dtype=bool)
    for one in preds:
        damage_total += one["damage_per_prey"]
        reported_any |= one["reported"]
    # The boolean is still what `resolve_deaths` needs: it asks whether predation damage occurred, not how
    # much. Only `update_health` needed the magnitude. DEV-031.
    damaged_any = damage_total > 0.0

    # 8, 9. pools.
    energy = update_energy(state["prey_energy"], sprinting, ate, params)
    stamina = update_stamina(state["prey_stamina"], sprinting, params)
    health = update_health(state["prey_health"], damage_total, energy, params)
    posture = update_posture(state["prey_posture"], action_executed)

    # 10. death, write-once.
    alive, cause, death_step, died_starving, died_predated = resolve_deaths(
        state["prey_alive"],
        health,
        energy,
        damaged_any,
        state["prey_death_cause"],
        state["prey_death_step"],
        state["prey_died_starving"],
        state["prey_died_predated"],
        step_index,
    )

    # 11. regrowth.
    food_grid, _ = regrow(
        entropy,
        food_grid,
        food_src,
        obstacles,
        params.cap_food,
        params.r_src,
        params.p_food,
        step_index,
        "regrow.food.trigger",
        "regrow.food.cell",
    )
    target_grid, _ = regrow(
        entropy,
        target_grid,
        target_src,
        obstacles,
        params.cap_target,
        params.r_src,
        params.p_target,
        step_index,
        "regrow.target.trigger",
        "regrow.target.cell",
    )

    # 12. accumulate. Online sums only.
    #
    # **Alive at the START of the tick, not after deaths resolve.** Written from the specification, not
    # transcribed from `src/env/`: a prey that dies this tick acted before capture resolved, so its action
    # belongs in the behavioural accumulators. `alive` stays the post-death mask everywhere else.
    #
    # Note that gate E2 cannot validate this change, because it is made identically on both sides and two
    # implementations changed the same way agree either way. What checks it is the prediction recorded with
    # the change: a prey dying at step d accumulates d alive ticks before and d+1 after.
    acting = np.asarray(state["prey_alive"], dtype=bool)

    engaged = np.zeros(k, dtype=bool)
    for i in range(k):
        for one in preds:
            if one["target"] == i and one["adjacent"] and one["engage"] >= params.e_min and bool(acting[i]):
                engaged[i] = True
    reported = reported_any
    unthreatened = (~reported) & acting
    emitted_stay = action_emitted == ACTION_STAY

    # ---- quantities the other four strategy classes need, spelled out per prey ------------
    # Written from the build spec's descriptor block, not transcribed from `src/env/`. FAR stands in for
    # "no predator" and "no living neighbour" and exceeds any distance the grid admits.
    h_, w_ = obstacles.shape
    FAR = h_ + w_
    n = len(alive)

    # Nearest predator, before and after this tick's movement. With no predators both are FAR, so `near` is
    # False and `dist_increased` is False, which is the correct reading of a predator-free world.
    def _nearest(point, positions):
        best = FAR
        for q in positions:
            best = min(best, chebyshev(point, (int(q[0]), int(q[1]))))
        return best

    prev_positions = [tuple(q) for q in state["pred_pos"]]
    now_positions = [tuple(one["pred_pos"]) for one in preds]

    visible_ticks = np.zeros(n, dtype=np.int32)
    flee_engaged = np.zeros(n, dtype=np.int32)
    visible_unengaged_ticks = np.zeros(n, dtype=np.int32)
    flee_visible_unengaged = np.zeros(n, dtype=np.int32)
    near_ticks = np.zeros(n, dtype=np.int32)
    occluded_near = np.zeros(n, dtype=np.int32)
    far_ticks = np.zeros(n, dtype=np.int32)
    occluded_far = np.zeros(n, dtype=np.int32)
    still_near = np.zeros(n, dtype=np.int32)
    accidental_still_near = np.zeros(n, dtype=np.int32)
    flee_near = np.zeros(n, dtype=np.int32)
    still_far = np.zeros(n, dtype=np.int32)
    flee_far = np.zeros(n, dtype=np.int32)
    away_near = np.zeros(n, dtype=np.int32)
    away_far = np.zeros(n, dtype=np.int32)
    nn_visible_ticks = np.zeros(n, dtype=np.int32)
    nn_unseen_ticks = np.zeros(n, dtype=np.int32)
    dist_sum_visible = np.zeros(n, dtype=np.float32)
    nn_sum_visible = np.zeros(n, dtype=np.float32)
    nn_sum_unseen = np.zeros(n, dtype=np.float32)
    density_sum = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if not bool(acting[i]):
            continue
        here = (int(prey_pos[i][0]), int(prey_pos[i][1]))
        there = (int(state["prey_pos"][i][0]), int(state["prey_pos"][i][1]))
        d_now = _nearest(here, now_positions)
        d_prev = _nearest(there, prev_positions)
        seen = bool(reported[i])
        near = d_now <= params.r_pred

        if seen:
            visible_ticks[i] += 1
            dist_sum_visible[i] += d_now
            if not bool(engaged[i]):
                visible_unengaged_ticks[i] += 1
                if d_now > d_prev:
                    flee_visible_unengaged[i] += 1
        if bool(engaged[i]) and d_now > d_prev:
            flee_engaged[i] += 1

        # Near-conditioned response, D-046. The build spec's events, conditioned on the geometric fact
        # rather than on the engagement flag, because the engagement flag is erased by the behaviours
        # these exist to measure. "Stayed" is the EMITTED action, per D-031; a blocked move that left the
        # prey in place is the accidental pathway and is counted separately.
        stayed = bool(emitted_stay[i])
        fled = d_now > d_prev
        # The emitted-direction flee event, D-048: the emitted move or sprint, executed from the
        # start-of-tick position against the predators' start-of-tick positions, would strictly increase
        # the nearest-predator distance. Intent rather than outcome, so pursuit cannot erase it.
        a = int(action_emitted[i])
        if 0 <= a <= 15:
            step = DIRECTIONS[a % 8]
            dest = (there[0] + step[0], there[1] + step[1])
            away = _nearest(dest, prev_positions) > d_prev
        else:
            away = False
        if near:
            near_ticks[i] += 1
            if not seen:
                occluded_near[i] += 1
            if stayed:
                still_near[i] += 1
            elif not bool(prey_moved[i]):
                accidental_still_near[i] += 1
            if fled:
                flee_near[i] += 1
            if away:
                away_near[i] += 1
        else:
            far_ticks[i] += 1
            if not seen:
                occluded_far[i] += 1
            if stayed:
                still_far[i] += 1
            if fled:
                flee_far[i] += 1
            if away:
                away_far[i] += 1

        # nearest living OTHER prey, and how many of them are within r_group
        best = FAR
        crowd = 0
        for j in range(n):
            if j == i or not bool(acting[j]):
                continue
            d = chebyshev(here, (int(prey_pos[j][0]), int(prey_pos[j][1])))
            best = min(best, d)
            if d <= params.r_group:
                crowd += 1
        density_sum[i] += crowd
        if best < FAR:
            if seen:
                nn_visible_ticks[i] += 1
                nn_sum_visible[i] += best
            else:
                nn_unseen_ticks[i] += 1
                nn_sum_unseen[i] += best

    acc = {
        "engaged_ticks": acc["engaged_ticks"] + engaged,
        "still_engaged": acc["still_engaged"] + (emitted_stay & engaged),
        "accidental_still": acc["accidental_still"] + (~emitted_stay & ~prey_moved & engaged),
        "unthreatened_ticks": acc["unthreatened_ticks"] + unthreatened,
        "still_unthreatened": acc["still_unthreatened"] + (emitted_stay & unthreatened),
        "alive_ticks": acc["alive_ticks"] + acting,
        "moved_ticks": acc["moved_ticks"] + prey_moved,
        "visible_ticks": acc["visible_ticks"] + visible_ticks,
        "flee_engaged": acc["flee_engaged"] + flee_engaged,
        "visible_unengaged_ticks": acc["visible_unengaged_ticks"] + visible_unengaged_ticks,
        "flee_visible_unengaged": acc["flee_visible_unengaged"] + flee_visible_unengaged,
        "near_ticks": acc["near_ticks"] + near_ticks,
        "occluded_near": acc["occluded_near"] + occluded_near,
        "far_ticks": acc["far_ticks"] + far_ticks,
        "occluded_far": acc["occluded_far"] + occluded_far,
        "dist_sum_visible": acc["dist_sum_visible"] + dist_sum_visible,
        "nn_sum_visible": acc["nn_sum_visible"] + nn_sum_visible,
        "nn_visible_ticks": acc["nn_visible_ticks"] + nn_visible_ticks,
        "nn_sum_unseen": acc["nn_sum_unseen"] + nn_sum_unseen,
        "nn_unseen_ticks": acc["nn_unseen_ticks"] + nn_unseen_ticks,
        "density_sum": acc["density_sum"] + density_sum,
        "still_near": acc["still_near"] + still_near,
        "accidental_still_near": acc["accidental_still_near"] + accidental_still_near,
        "flee_near": acc["flee_near"] + flee_near,
        "still_far": acc["still_far"] + still_far,
        "flee_far": acc["flee_far"] + flee_far,
        "away_near": acc["away_near"] + away_near,
        "away_far": acc["away_far"] + away_far,
    }

    new_state = {
        "prey_pos": prey_pos,
        "prey_alive": alive,
        "prey_energy": energy,
        "prey_stamina": stamina,
        "prey_health": health,
        "prey_posture": posture,
        "prey_hidden": hidden,
        "prey_score": prey_score,
        "prey_moved": prey_moved,
        "prey_death_cause": cause,
        "prey_death_step": death_step,
        "prey_died_starving": died_starving,
        "prey_died_predated": died_predated,
        # Fixed at reset and carried unchanged, exactly as the implementation carries it. Without this the
        # field exists at tick 0 and vanishes at tick 1, which the field-set assertion catches immediately.
        "spawn_relax_level": state["spawn_relax_level"],
        "pred_pos": _stack_pred(preds, "pred_pos", (2,), np.int32),
        "pred_stamina": _stack_pred(preds, "stamina", (), np.float32),
        "pred_target": _stack_pred(preds, "target", (), np.int32),
        "pred_engage": _stack_pred(preds, "engage", (), np.int32),
        "pred_still": _stack_pred(preds, "still", (), np.float32),
        "pred_appearance": state["pred_appearance"],
        "food_grid": food_grid,
        "target_grid": target_grid,
    }
    # **The record carries what the RENDERER and the trajectory tier need, which is more than the
    # descriptors need.** D-034 set that schema on the implementation side; the oracle carried only four
    # of its nine fields until queue item E, so `prey_pos` and `pred_pos`, the two the trajectory tier
    # reads as ground truth, had never been differentially compared. Every value here is one this
    # function already computed for `new_state`, exposed under the record's name rather than derived
    # again, which keeps the oracle independent of the implementation's code while making the two
    # comparable.
    record = {
        "prey_pos": prey_pos,
        "prey_alive": alive,
        "prey_posture": posture,
        "prey_moved": prey_moved,
        "pred_pos": _stack_pred(preds, "pred_pos", (2,), np.int32),
        "food_grid": food_grid,
        "target_grid": target_grid,
        "action_emitted": action_emitted,
        "action_executed": action_executed,
    }
    return new_state, acc, record


def _stack_pred(preds: list, key: str, shape: tuple, dtype) -> np.ndarray:
    """Stack one field across predators, preserving the leading axis at zero predators.

    `np.array([])` would give shape (0,) rather than (0, 2), and a shape mismatch at `n_pred = 0` is exactly
    the silent kind: it would surface as a broadcasting surprise somewhere else entirely.
    """
    out = np.zeros((len(preds), *shape), dtype=dtype)
    for j, one in enumerate(preds):
        out[j] = one[key]
    return out


def reset_derived(prey_pos, pred_pos, obstacles, food_grid, target_grid, appearance, params) -> dict:
    """Everything `reset` DERIVES, given the positions it drew. Written from the build specification.

    **Positions are injected rather than drawn here, and that limit is the point of DEV-022.** `reset` in the
    implementation calls `jax.random` directly at three draw sites, so a NumPy oracle cannot reproduce the
    positions: threefry and PCG64 give different streams from the same seed. Until those draws go through the
    tagged entropy source, the positions stay shared driver state, which DEV-021 shows is upstream of the fork
    and invisible to the differential.

    What this DOES buy: the relaxation level, the pools, the hidden state and the death records stop being
    shared and start being independently derived. That empties `UNMODELLED` and moves the fork by the size of
    those quantities. It does not move it for the positions, and describing it as though it did would be
    exactly the overclaim DEV-021 warns about.

    The relaxation ladder is recomputed here from the prey positions and the obstacle grid, which is the one
    quantity E2 could not compare at all before.
    """
    h, w = obstacles.shape
    k = len(prey_pos)
    n_pred = len(pred_pos)

    # The relaxation level: the first rung with at least `n_pred` free cells at that separation from every
    # prey, or -1 if no rung qualifies. Spelled out rather than vectorised, as everywhere in this file.
    level = -1
    for i, d in enumerate(params.d_spawn_relax):
        count = 0
        for y in range(h):
            for x in range(w):
                if bool(obstacles[y, x]):
                    continue
                nearest = min(max(abs(y - int(p[0])), abs(x - int(p[1]))) for p in prey_pos) if k else h + w
                if nearest >= d:
                    count += 1
        if count >= n_pred:
            level = i
            break

    return {
        "prey_pos": np.array(prey_pos, dtype=np.int32),
        "prey_alive": np.ones(k, dtype=bool),
        "prey_energy": np.full(k, params.e_max, dtype=np.float32),
        "prey_stamina": np.full(k, params.s_max, dtype=np.float32),
        "prey_health": np.full(k, params.h_max, dtype=np.float32),
        "prey_posture": np.zeros(k, dtype=np.int32),
        # **Exactly zero, and never inherited.** Invariant I3, and gate S8 checks it behaviourally.
        "prey_hidden": np.zeros((k, params.hd), dtype=np.float32),
        "prey_score": np.zeros(k, dtype=np.int32),
        "prey_moved": np.zeros(k, dtype=bool),
        "prey_death_cause": np.zeros(k, dtype=np.int32),
        "prey_death_step": np.full(k, -1, dtype=np.int32),
        "prey_died_starving": np.zeros(k, dtype=bool),
        "prey_died_predated": np.zeros(k, dtype=bool),
        "pred_pos": np.array(pred_pos, dtype=np.int32).reshape(n_pred, 2),
        "pred_stamina": np.full(n_pred, params.s_max, dtype=np.float32),
        "pred_target": np.full(n_pred, NO_TARGET, dtype=np.int32),
        "pred_engage": np.zeros(n_pred, dtype=np.int32),
        "pred_still": np.zeros(n_pred, dtype=np.float32),
        "pred_appearance": np.array(appearance, dtype=np.float32).reshape(n_pred, params.a_appearance),
        "food_grid": np.array(food_grid, dtype=np.int32),
        "target_grid": np.array(target_grid, dtype=np.int32),
        "spawn_relax_level": np.int32(level),
    }


def zero_accumulators(k: int) -> dict:
    """Counts int32, running sums float32, matching the implementation's split.

    The four classes below `moved_ticks` arrived at step 8. Until then the accumulator set served freezing
    completely and fleeing, occlusion, distance maintenance and grouping not at all. DEV-019.
    """
    counts = (
        "engaged_ticks",
        "still_engaged",
        "accidental_still",
        "unthreatened_ticks",
        "still_unthreatened",
        "alive_ticks",
        "moved_ticks",
        "visible_ticks",
        "flee_engaged",
        "visible_unengaged_ticks",
        "flee_visible_unengaged",
        "near_ticks",
        "occluded_near",
        "far_ticks",
        "occluded_far",
        "nn_visible_ticks",
        "nn_unseen_ticks",
        # near-conditioned response, D-046: geometric conditioning the behaviour cannot erase
        "still_near",
        "accidental_still_near",
        "flee_near",
        "still_far",
        "flee_far",
        # the emitted-direction flee event, D-048: intent, which pursuit cannot erase
        "away_near",
        "away_far",
    )
    sums = ("dist_sum_visible", "nn_sum_visible", "nn_sum_unseen", "density_sum")
    out = {n: np.zeros(k, dtype=np.int32) for n in counts}
    out.update({n: np.zeros(k, dtype=np.float32) for n in sums})
    return out
