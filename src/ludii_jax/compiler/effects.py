"""
Effect compilers: capture, flip, promote, score updates.

Each returns a function: (state, original_player) → state.
"""

import jax
import jax.numpy as jnp

from ..runtime.state import BOARD_DTYPE, ACTION_DTYPE, REWARD_DTYPE, EMPTY


def compile_custodial_capture(topology, adjacency_lookup, piece_idx, length=1, num_players=2):
    """Custodial capture: remove enemy pieces sandwiched between friendly pieces.

    Vectorized: precompute all (endpoint1, middle, endpoint2) triples as arrays,
    then check all at once with JAX indexing.
    """
    import numpy as np
    n = topology.num_sites

    # Precompute custodial triples: (endpoint1, middle, endpoint2) for length=1
    endpoints1 = []
    middles = []
    endpoints2 = []
    for d in range(topology.max_neighbors):
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

    def apply_fn(state, original_player):
        mover_mask = (state.board == original_player).any(axis=0)
        enemy_mask = ((state.board != EMPTY) & (state.board != original_player)).any(axis=0)

        # Vectorized check: both endpoints friendly, middle is enemy
        is_custodial = mover_mask[ep1] & mover_mask[ep2] & enemy_mask[mid]

        # Scatter captures: for each triggered triple, mark the middle cell
        capture_mask = jnp.zeros(n, dtype=jnp.bool_)
        capture_mask = capture_mask.at[mid].set(capture_mask[mid] | is_custodial)

        board = jnp.where(capture_mask[jnp.newaxis, :], EMPTY, state.board)
        return state._replace(board=board)

    return apply_fn


def compile_surround_capture(topology, corner_only=True, num_players=2):
    """Surround capture: remove enemy in corner when all neighbors are friendly.

    For Hasami Shogi: enemy in corner is captured when all its orthogonal
    neighbors are friendly pieces or board edges.
    """
    import numpy as np
    n = topology.num_sites

    # Find corner cells
    if corner_only:
        corner_cells = []
        for i in range(n):
            nb_count = sum(1 for d in range(topology.max_neighbors) if int(topology.adjacency[d, i]) < n)
            if nb_count <= 2:  # corner = 2 or fewer neighbors
                corner_cells.append(i)
        target_cells = jnp.array(corner_cells, dtype=ACTION_DTYPE) if corner_cells else jnp.array([], dtype=ACTION_DTYPE)
    else:
        target_cells = jnp.arange(n, dtype=ACTION_DTYPE)

    # Precompute neighbor lists per cell (padded)
    max_nb = topology.max_neighbors
    adj = jnp.array(topology.adjacency)

    def apply_fn(state, original_player):
        mover_mask = (state.board == original_player).any(axis=0)
        enemy_mask = ((state.board != EMPTY) & (state.board != original_player)).any(axis=0)

        capture_mask = jnp.zeros(n, dtype=jnp.bool_)
        for cell in corner_cells if corner_only else range(n):
            is_enemy = enemy_mask[cell]
            # Check all neighbors: each must be friendly or off-board
            all_blocked = jnp.bool_(True)
            for d in range(max_nb):
                nb = adj[d, cell]
                nb_ok = (nb >= n) | mover_mask[nb.clip(0, n - 1)]  # off-board or friendly
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
        scores = state.scores.at[original_player].set(count)
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
