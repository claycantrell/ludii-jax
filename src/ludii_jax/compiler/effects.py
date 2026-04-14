"""
Effect compilers: capture, flip, promote, score updates.

Each returns a function: (state, original_player) → state.
"""

import jax
import jax.numpy as jnp

from ..runtime.state import BOARD_DTYPE, ACTION_DTYPE, REWARD_DTYPE, EMPTY


def compile_custodial_capture(topology, adjacency_lookup, piece_idx, length=1, num_players=2):
    """Custodial capture: remove enemy pieces sandwiched between friendly pieces."""
    n = topology.num_sites

    # Precompute custodial line indices
    lines = []
    for d in range(topology.max_neighbors):
        for site in range(n):
            line = [site]
            pos = site
            for _ in range(length + 1):
                next_pos = int(topology.adjacency[d, pos])
                if next_pos >= n:
                    break
                line.append(next_pos)
                pos = next_pos
            if len(line) >= length + 2:
                lines.append(line[:length + 2])

    if not lines:
        return lambda state, op: state

    line_arr = jnp.array(lines, dtype=ACTION_DTYPE)

    def apply_fn(state, original_player):
        occupied_mover = (state.board == original_player).any(axis=0)
        occupied_enemy = ((state.board != EMPTY) & (state.board != original_player)).any(axis=0)

        # Check each potential custodial line
        capture_mask = jnp.zeros(n, dtype=jnp.bool_)
        for line in lines:
            endpoints_mine = occupied_mover[line[0]] & occupied_mover[line[-1]]
            middle_enemy = jnp.array([occupied_enemy[line[i]] for i in range(1, len(line) - 1)]).all()
            is_custodial = endpoints_mine & middle_enemy
            for i in range(1, len(line) - 1):
                capture_mask = capture_mask | (is_custodial & (jnp.arange(n) == line[i]))

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
