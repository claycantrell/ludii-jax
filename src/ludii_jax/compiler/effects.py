"""
Effect compilers: capture, flip, promote, score updates.

Each returns a function: (state, original_player) → state.
"""

import jax
import jax.numpy as jnp

from ..runtime.state import BOARD_DTYPE, ACTION_DTYPE, REWARD_DTYPE, EMPTY


def compile_custodial_capture(topology, adjacency_lookup, piece_idx, length=1, num_players=2,
                               directions=None, hostile_cells=None):
    """Custodial capture: remove enemy pieces sandwiched between friendly pieces.

    Vectorized: precompute all (endpoint1, middle, endpoint2) triples as arrays,
    then check all at once with JAX indexing.
    directions: list of direction indices to check (None = all).
    hostile_cells: optional bool array (n,) — cells that act as friendly endpoints
    even when empty (e.g. throne in Tablut).
    """
    import numpy as np
    n = topology.num_sites
    allowed_dirs = set(range(topology.max_neighbors)) if directions is None else set(directions)

    # Precompute custodial triples: (endpoint1, middle, endpoint2) for length=1
    endpoints1 = []
    middles = []
    endpoints2 = []
    for d in range(topology.max_neighbors):
        if d not in allowed_dirs:
            continue
        for site in range(n):
            pos = site
            line = [site]
            for _ in range(length + 1):
                next_pos = int(topology.adjacency[d, pos])
                if next_pos >= n:
                    break
                line.append(next_pos)
                pos = next_pos
            if len(line) >= length + 2:
                endpoints1.append(line[0])
                middles.append(line[1])  # for length=1, single middle
                endpoints2.append(line[-1])

    if not endpoints1:
        return lambda state, op: state

    ep1 = jnp.array(endpoints1, dtype=ACTION_DTYPE)
    mid = jnp.array(middles, dtype=ACTION_DTYPE)
    ep2 = jnp.array(endpoints2, dtype=ACTION_DTYPE)

    _hostile = hostile_cells if hostile_cells is not None else jnp.zeros(n, dtype=jnp.bool_)

    def apply_fn(state, original_player):
        # Hostile cells (e.g. throne) count as friendly endpoints
        mover_mask = (state.board == original_player).any(axis=0) | _hostile
        enemy_mask = ((state.board != EMPTY) & (state.board != original_player)).any(axis=0)

        # Only check triples radiating from the last-moved-to cell
        last_to = state.previous_actions[num_players]
        from_last = (ep1 == last_to) | (ep2 == last_to)

        # Vectorized check: both endpoints friendly, middle is enemy, involves last move
        is_custodial = mover_mask[ep1] & mover_mask[ep2] & enemy_mask[mid] & from_last

        capture_mask = jnp.zeros(n, dtype=jnp.int8)
        capture_mask = capture_mask.at[mid].max(is_custodial.astype(jnp.int8))
        capture_mask = capture_mask.astype(jnp.bool_)

        board = jnp.where(capture_mask[jnp.newaxis, :], EMPTY, state.board)
        return state._replace(board=board)

    return apply_fn


def compile_surround_capture(topology, corner_only=True, num_players=2, directions=None):
    """Surround capture: remove enemy when all specified neighbors are friendly/edge.

    For Hasami Shogi: enemy in corner captured when all orthogonal neighbors
    are friendly pieces or board edges.
    """
    n = topology.num_sites
    max_nb = topology.max_neighbors
    check_dirs = list(range(max_nb)) if directions is None else directions

    # Find corner cells: cells with fewest orthogonal neighbors
    if corner_only:
        nb_counts = [sum(1 for d in check_dirs if int(topology.adjacency[d, i]) < n) for i in range(n)]
        min_nb = min(nb_counts) if nb_counts else 0
        corner_cells = [i for i in range(n) if nb_counts[i] == min_nb]
    else:
        corner_cells = list(range(n))

    if not corner_cells:
        return lambda state, op: state

    # Precompute per-corner: which directions have real neighbors
    adj = jnp.array(topology.adjacency)

    def apply_fn(state, original_player):
        mover_mask = (state.board == original_player).any(axis=0)
        enemy_mask = ((state.board != EMPTY) & (state.board != original_player)).any(axis=0)

        capture_mask = jnp.zeros(n, dtype=jnp.bool_)
        for cell in corner_cells:
            is_enemy = enemy_mask[cell]
            all_blocked = jnp.bool_(True)
            for d in check_dirs:
                nb = adj[d, cell]
                nb_ok = (nb >= n) | mover_mask[nb.clip(0, n - 1)]
                all_blocked = all_blocked & nb_ok
            capture_mask = capture_mask.at[cell].set(is_enemy & all_blocked)

        board = jnp.where(capture_mask[jnp.newaxis, :], EMPTY, state.board)
        return state._replace(board=board)

    return apply_fn


def compile_flip(topology, piece_idx, num_players=2):
    """Flip effect: change piece ownership (Reversi-style)."""
    n = topology.num_sites

    def apply_fn(state, original_player):
        # Flip pieces adjacent to the last move that are sandwiched
        # Simplified: flip all adjacent enemy pieces
        last_pos = state.previous_actions[num_players]
        adj = jnp.array(topology.adjacency)

        # For each direction, check if we can flip
        board = state.board
        for d in range(topology.max_neighbors):
            nb = adj[d, last_pos]
            is_enemy = jnp.where(nb < n,
                                  ((board[:, nb] != EMPTY) & (board[:, nb] != original_player)).any(),
                                  False)
            board = jnp.where(is_enemy & (jnp.arange(board.shape[0])[:, jnp.newaxis] == piece_idx),
                              jnp.where(jnp.arange(n)[jnp.newaxis, :] == nb, original_player, board),
                              board)

        return state._replace(board=board)

    return apply_fn


def compile_set_score(num_players):
    """Set score to count of occupied cells."""
    def apply_fn(state, original_player):
        count = (state.board == original_player).any(axis=0).sum().astype(REWARD_DTYPE)
        scores = state.scores.at[original_player.astype(jnp.int32)].set(count)
        return state._replace(scores=scores)
    return apply_fn


def compile_extra_turn(num_players):
    """Grant an extra turn to the current player."""
    def apply_fn(state, original_player):
        return state._replace(
            phase_step_count=jnp.maximum(state.phase_step_count - 1, 0),
            current_player=original_player
        )
    return apply_fn


def chain_effects(effect_fns):
    """Chain multiple effects into one function."""
    if not effect_fns:
        return lambda state, op: state
    if len(effect_fns) == 1:
        return effect_fns[0]

    def combined(state, original_player):
        for fn in effect_fns:
            state = fn(state, original_player)
        return state
    return combined
