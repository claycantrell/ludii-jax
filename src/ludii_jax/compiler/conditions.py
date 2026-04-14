"""
End condition compilers: line, connected, no moves, full board, captured all.

Each returns a function: state → (winners, terminated).
"""

import jax
import jax.numpy as jnp

from ..runtime.state import BOARD_DTYPE, EMPTY


def compile_line_win(line_indices, piece_idx, num_players, exclude_regions=None):
    """Win by forming a line of N.

    exclude_regions: optional (num_players, n) bool mask. If set, lines entirely
    within the current player's region don't count (e.g. starting rows).
    """
    if line_indices.shape[0] == 0:
        return lambda state: (EMPTY * jnp.ones(num_players, jnp.int8), False)

    def end_fn(state):
        occupied = (state.board[piece_idx] == state.current_player).astype(BOARD_DTYPE)
        line_matches = (occupied[line_indices] == 1).all(axis=1)
        if exclude_regions is not None:
            # Exclude lines where ALL cells are in the mover's starting region
            player_region = exclude_regions[state.current_player]
            all_in_region = player_region[line_indices].all(axis=1)
            line_matches = line_matches & ~all_in_region
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


def compile_captured_all(num_players, min_pieces=3):
    """Win when opponent has fewer than min_pieces (default 3, like NMM).
    Only triggers after opponent has HAD at least min_pieces (captures happened)."""
    def end_fn(state):
        opponent = (state.current_player + 1) % num_players
        opp_count = (state.board == opponent).any(axis=0).sum()
        mover_count = (state.board == state.current_player).any(axis=0).sum()
        total = opp_count + mover_count
        # Only trigger after enough pieces have been placed (both players active)
        # and opponent count has dropped below threshold from captures
        won = (opp_count < min_pieces) & (total >= 2 * min_pieces)
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


def compile_connected_win(topology, side_sets, piece_idx, num_players):
    """Win by connecting specified side sets through mover's pieces.

    side_sets: list of (set_a, set_b) pairs. Win if any pair is connected.
    Uses iterative flood-fill with jax.lax.fori_loop for efficient tracing.
    """
    import numpy as np_cpu
    n = topology.num_sites
    adj = jnp.array(topology.adjacency)
    max_nb = topology.max_neighbors
    max_iters = min(n, 20)

    # Precompute neighbor lookup as flat array: for each cell, list of valid neighbors
    # Shape: (n, max_nb) padded with n for invalid
    nb_table = jnp.array(topology.adjacency.T.clip(0, n), dtype=jnp.int32)  # (n, max_nb)
    nb_valid = jnp.array((topology.adjacency < n).T, dtype=jnp.bool_)  # (n, max_nb)

    pairs = []
    for set_a, set_b in side_sets:
        mask_a = jnp.zeros(n, dtype=jnp.bool_)
        mask_b = jnp.zeros(n, dtype=jnp.bool_)
        for c in set_a: mask_a = mask_a.at[c].set(True)
        for c in set_b: mask_b = mask_b.at[c].set(True)
        pairs.append((mask_a, mask_b))

    def _flood_fill(occupied, seed):
        """BFS flood fill from seed through occupied cells."""
        def step(_, visited):
            # For each visited cell, mark its neighbors as reachable
            expanded = visited[jnp.arange(n), jnp.newaxis] & nb_valid  # (n, max_nb)
            nb_reached = jnp.zeros(n, dtype=jnp.bool_)
            # Scatter: for each cell's neighbors, OR into nb_reached
            nb_reached = nb_reached.at[nb_table.flatten()].max(
                expanded.flatten().astype(jnp.int8)).astype(jnp.bool_)
            return visited | (nb_reached & occupied)
        return jax.lax.fori_loop(0, max_iters, step, seed & occupied)

    def end_fn(state):
        occupied = (state.board[piece_idx] == state.current_player)
        won = jnp.bool_(False)
        for mask_a, mask_b in pairs:
            reached = _flood_fill(occupied, mask_a)
            won = won | (reached & mask_b).any()
        winners = jnp.where(won,
                            jnp.zeros(num_players, jnp.int8).at[state.current_player].set(1),
                            EMPTY * jnp.ones(num_players, jnp.int8))
        return winners, won

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
