"""
End condition compilers: line, connected, no moves, full board, captured all.

Each returns a function: state → (winners, terminated).
"""

import jax
import jax.numpy as jnp

from ..runtime.state import BOARD_DTYPE, EMPTY


def compile_line_win(line_indices, piece_idx, num_players):
    """Win by forming a line of N."""
    if line_indices.shape[0] == 0:
        return lambda state: (EMPTY * jnp.ones(num_players, jnp.int8), False)

    def end_fn(state):
        occupied = (state.board[piece_idx] == state.current_player).astype(BOARD_DTYPE)
        line_matches = (occupied[line_indices] == 1).all(axis=1)
        won = line_matches.any()
        winners = jnp.where(won,
                            jnp.zeros(num_players, jnp.int8).at[state.current_player].set(1),
                            EMPTY * jnp.ones(num_players, jnp.int8))
        return winners, won

    return end_fn


def compile_line_loss(line_indices, piece_idx, num_players):
    """Lose by forming a line of N."""
    if line_indices.shape[0] == 0:
        return lambda state: (EMPTY * jnp.ones(num_players, jnp.int8), False)

    def end_fn(state):
        occupied = (state.board[piece_idx] == state.current_player).astype(BOARD_DTYPE)
        line_matches = (occupied[line_indices] == 1).all(axis=1)
        lost = line_matches.any()
        winners = jnp.where(lost,
                            jnp.zeros(num_players, jnp.int8).at[(state.current_player + 1) % num_players].set(1),
                            EMPTY * jnp.ones(num_players, jnp.int8))
        return winners, lost

    return end_fn


def compile_no_moves_loss(num_players):
    """Lose when you have no legal moves."""
    def end_fn(state):
        no_moves = ~state.legal_action_mask.any()
        # Current player (who just moved) wins, opponent (next) loses
        winners = jnp.where(no_moves,
                            jnp.zeros(num_players, jnp.int8).at[state.current_player].set(1),
                            EMPTY * jnp.ones(num_players, jnp.int8))
        return winners, no_moves
    return end_fn


def compile_captured_all(num_players):
    """Win when opponent has no pieces left."""
    def end_fn(state):
        opponent = (state.current_player + 1) % num_players
        opponent_has_pieces = (state.board == opponent).any()
        won = ~opponent_has_pieces
        winners = jnp.where(won,
                            jnp.zeros(num_players, jnp.int8).at[state.current_player].set(1),
                            EMPTY * jnp.ones(num_players, jnp.int8))
        return winners, won
    return end_fn


def compile_full_board_draw(num_players):
    """Draw when the board is full."""
    def end_fn(state):
        full = (state.board != EMPTY).any(axis=0).all()
        winners = jnp.where(full, EMPTY * jnp.ones(num_players, jnp.int8),
                            EMPTY * jnp.ones(num_players, jnp.int8))
        return winners, full
    return end_fn


def compile_full_board_by_score(num_players):
    """Highest score wins when board is full."""
    def end_fn(state):
        full = (state.board != EMPTY).any(axis=0).all()
        best = jnp.argmax(state.scores)
        winners = jnp.where(full,
                            jnp.zeros(num_players, jnp.int8).at[best].set(1),
                            EMPTY * jnp.ones(num_players, jnp.int8))
        return winners, full
    return end_fn


def combine_end_conditions(end_fns, num_players):
    """Combine multiple end conditions. First one that fires wins."""
    if not end_fns:
        return lambda state: (EMPTY * jnp.ones(num_players, jnp.int8), False)

    def combined(state):
        for fn in end_fns:
            winners, ended = fn(state)
            if ended:
                return winners, ended
        return EMPTY * jnp.ones(num_players, jnp.int8), False

    # JAX-compatible version using lax
    def combined_jax(state):
        all_results = [fn(state) for fn in end_fns]
        all_winners = jnp.stack([w for w, _ in all_results])
        all_ended = jnp.array([e for _, e in all_results])
        first_active = jnp.argmax(all_ended)
        any_ended = all_ended.any()
        winners = jax.lax.select(any_ended, all_winners[first_active],
                                  EMPTY * jnp.ones(num_players, jnp.int8))
        return winners, any_ended

    return combined_jax
