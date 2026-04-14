"""
Move compilers: ludeme → JAX functions.

Each compile_* function takes structured parameters (topology, piece index,
direction mask, etc.) and returns (legal_action_fn, apply_action_fn).

All movement produces (num_sites, num_sites) FROM_TO masks.
action = source * num_sites + destination.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..runtime.state import BOARD_DTYPE, ACTION_DTYPE, EMPTY


def compile_step(topology, slide_lookup, piece_idx, num_players, distance=1):
    """Compile step movement: move exactly `distance` cells."""
    n = topology.num_sites
    max_nb = topology.max_neighbors
    adj = jnp.array(topology.adjacency)
    arange_n = jnp.arange(n, dtype=jnp.int32)

    def legal_fn(state):
        owned = (state.board[piece_idx] == state.current_player)
        friendly = (state.board == state.current_player).any(axis=0)
        # Can step to empty OR enemy cells (not friendly)
        dests = slide_lookup[:, :, distance]  # (max_nb, n)
        on_board = dests < n
        dest_not_friendly = ~friendly.at[dests.clip(0, n - 1)].get()
        valid = owned[jnp.newaxis, :] & dest_not_friendly & on_board  # (max_nb, n)
        # Scatter to flat mask
        flat_idx = arange_n[jnp.newaxis, :] * n + dests.clip(0, n - 1)
        mask = jnp.zeros(n * n, dtype=BOARD_DTYPE)
        mask = mask.at[flat_idx.flatten()].set(valid.flatten().astype(BOARD_DTYPE))
        return mask

    def apply_fn(state, action):
        src, dst = action // n, action % n
        board = state.board.at[:, dst].set(EMPTY)  # clear destination (capture)
        board = board.at[piece_idx, src].set(EMPTY)
        board = board.at[piece_idx, dst].set(state.current_player)
        pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(dst)).at[num_players].set(ACTION_DTYPE(dst))
        return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_hop(topology, slide_lookup, hop_between, piece_idx, num_players,
                hop_over="opponent", capture=True):
    """Compile hop movement: jump over a piece to land beyond it."""
    n = topology.num_sites
    max_nb = topology.max_neighbors
    arange_n = jnp.arange(n, dtype=jnp.int32)
    hop_over_friendly = (hop_over == "mover")

    def legal_fn(state):
        owned = (state.board[piece_idx] == state.current_player)
        if hop_over_friendly:
            hop_mask = (state.board == state.current_player).any(axis=0)
        else:
            hop_mask = ((state.board != EMPTY) & (state.board != state.current_player)).any(axis=0)

        between = slide_lookup[:, :, 1]  # (max_nb, n)
        dests = slide_lookup[:, :, 2]    # (max_nb, n)
        on_board = dests < n
        has_hop_piece = hop_mask.at[between.clip(0, n - 1)].get()
        valid = owned[jnp.newaxis, :] & on_board & has_hop_piece.astype(jnp.bool_)

        if hop_over_friendly:
            enemy = ((state.board != EMPTY) & (state.board != state.current_player)).any(axis=0)
            valid = valid & enemy.at[dests.clip(0, n - 1)].get()
        else:
            occupied = (state.board != EMPTY).any(axis=0)
            valid = valid & ~occupied.at[dests.clip(0, n - 1)].get()

        flat_idx = arange_n[jnp.newaxis, :] * n + dests.clip(0, n - 1)
        mask = jnp.zeros(n * n, dtype=BOARD_DTYPE)
        mask = mask.at[flat_idx.flatten()].set(valid.flatten().astype(BOARD_DTYPE))
        return mask

    if capture:
        def apply_fn(state, action):
            src, dst = action // n, action % n
            between_cell = hop_between[src, dst]
            board = state.board.at[piece_idx, src].set(EMPTY)
            if hop_over_friendly:
                board = board.at[:, dst].set(EMPTY)
                board = board.at[piece_idx, dst].set(state.current_player)
            else:
                board = board.at[piece_idx, dst].set(state.current_player)
                board = board.at[:, between_cell].set(EMPTY)
            pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(dst)).at[num_players].set(ACTION_DTYPE(dst))
            return state._replace(board=board, previous_actions=pa)
    else:
        def apply_fn(state, action):
            src, dst = action // n, action % n
            board = state.board.at[piece_idx, dst].set(state.current_player)
            board = board.at[piece_idx, src].set(EMPTY)
            pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(dst)).at[num_players].set(ACTION_DTYPE(dst))
            return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_slide(topology, slide_lookup, piece_idx, num_players, max_distance=None):
    """Compile slide movement: move any number of cells in a direction until blocked."""
    n = topology.num_sites
    max_nb = topology.max_neighbors
    if max_distance is None:
        max_distance = min(n, 32)
    arange_n = jnp.arange(n, dtype=ACTION_DTYPE)
    general_idx = jnp.indices((n, max_nb, max_distance), dtype=ACTION_DTYPE)[2]
    occupied_pad = jnp.ones((n, max_nb, 1), dtype=ACTION_DTYPE)

    def legal_fn(state):
        occupied = (state.board != EMPTY).any(axis=0).astype(BOARD_DTYPE)
        slide_indices = slide_lookup[:, :, :max_distance].transpose(1, 0, 2)
        occupied_at = occupied.at[slide_indices].get(mode="fill", fill_value=1)
        occupied_at = occupied_at.at[:, :, 0].set(0)
        occupied_at = jnp.concatenate([occupied_at, occupied_pad], axis=2)
        slide_until = jnp.argmax(occupied_at, axis=2)

        valid_dests = jnp.where(
            general_idx < slide_until[:, :, jnp.newaxis],
            slide_indices, n + 1
        ).reshape(n, -1)

        mask = jnp.zeros((n, n), dtype=BOARD_DTYPE)
        mask = mask.at[arange_n[:, jnp.newaxis], valid_dests].set(1)
        mask = mask.at[arange_n, arange_n].set(0)

        piece_mask = (state.board[piece_idx] == state.current_player).astype(BOARD_DTYPE)
        mask = jnp.where(piece_mask[:, jnp.newaxis], mask, jnp.zeros_like(mask))
        return mask.flatten()

    def apply_fn(state, action):
        src, dst = action // n, action % n
        board = state.board.at[piece_idx, dst].set(state.current_player)
        board = board.at[piece_idx, src].set(EMPTY)
        pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(dst)).at[num_players].set(ACTION_DTYPE(dst))
        return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_leap(topology, piece_idx, num_players, offsets, capture=False):
    """Compile leap movement: non-adjacent jumps (knight, camel, etc.)."""
    from ..runtime.lookup import build_line_indices  # avoid circular
    n = topology.num_sites

    # Precompute leap destinations from offsets
    leap_lookup = np.full((len(offsets), n), n, dtype=np.int16)
    coords = topology.site_coords
    for oi, (dr, dc) in enumerate(offsets):
        for cell in range(n):
            # Find nearest site to (x+dc, y+dr)
            cx, cy = coords[cell]
            target_x, target_y = cx + dc, cy + dr
            best_dist = float('inf')
            best_idx = n
            for j in range(n):
                jx, jy = coords[j]
                d = (jx - target_x) ** 2 + (jy - target_y) ** 2
                if d < best_dist and d < 0.5:  # must be very close
                    best_dist = d
                    best_idx = j
            leap_lookup[oi, cell] = best_idx

    leap_lookup = jnp.array(leap_lookup, dtype=ACTION_DTYPE)
    arange_n = jnp.arange(n, dtype=jnp.int32)

    def legal_fn(state):
        owned = (state.board[piece_idx] == state.current_player)
        friendly = (state.board == state.current_player).any(axis=0)
        mask = jnp.zeros(n * n, dtype=BOARD_DTYPE)
        for oi in range(len(offsets)):
            dests = leap_lookup[oi]
            valid = (dests < n) & owned & ~friendly.at[dests.clip(0, n - 1)].get()
            flat_idx = arange_n * n + dests.clip(0, n - 1)
            mask = mask.at[flat_idx].add(valid.astype(BOARD_DTYPE))
        return (mask > 0).astype(BOARD_DTYPE)

    if capture:
        def apply_fn(state, action):
            src, dst = action // n, action % n
            board = state.board.at[piece_idx, src].set(EMPTY)
            board = board.at[:, dst].set(EMPTY)
            board = board.at[piece_idx, dst].set(state.current_player)
            pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(dst)).at[num_players].set(ACTION_DTYPE(dst))
            return state._replace(board=board, previous_actions=pa)
    else:
        def apply_fn(state, action):
            src, dst = action // n, action % n
            board = state.board.at[piece_idx, dst].set(state.current_player)
            board = board.at[piece_idx, src].set(EMPTY)
            pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(dst)).at[num_players].set(ACTION_DTYPE(dst))
            return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_place(topology, piece_idx, num_players):
    """Compile placement: place a piece on any empty site."""
    n = topology.num_sites

    def legal_fn(state):
        return (state.board == EMPTY).all(axis=0).astype(BOARD_DTYPE)

    def apply_fn(state, action):
        board = state.board.at[piece_idx, action].set(state.current_player)
        pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(action)).at[num_players].set(ACTION_DTYPE(action))
        return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_sow(topology, num_players, initial_seeds=4, has_stores=True):
    """Compile mancala sowing: pick a pit, distribute seeds along track."""
    n = topology.num_sites
    # For mancala with stores: stores are non-sowable cells
    # Standard layout: P1 pits, P1 store, P2 pits, P2 store
    if has_stores and n >= 4:
        pits_per_player = (n - 2) // 2
        p1_store = pits_per_player
        p2_store = n - 1
        is_store = jnp.zeros(n, dtype=jnp.bool_).at[p1_store].set(True).at[p2_store].set(True)
    else:
        pits_per_player = n // 2
        is_store = jnp.zeros(n, dtype=jnp.bool_)

    half = n // 2

    # Build counter-clockwise track
    if n > 2:
        w = n // 2 if n % 2 == 0 else n
        track = list(range(w)) + list(range(n - 1, w - 1, -1)) if n >= 4 else list(range(n))
    else:
        track = list(range(n))
    track_arr = jnp.array(track[:n], dtype=ACTION_DTYPE)
    track_len = len(track_arr)

    track_pos = jnp.full(n, 0, dtype=ACTION_DTYPE)
    for i, cell in enumerate(track[:n]):
        track_pos = track_pos.at[cell].set(i)

    pit_owner_init = jnp.concatenate([
        jnp.zeros(half, dtype=BOARD_DTYPE),
        jnp.ones(n - half, dtype=BOARD_DTYPE)
    ])

    def legal_fn(state):
        owned = (state.pit_owner == state.current_player)
        has_seeds = (state.seed_counts > 0)
        not_store = ~is_store
        return (owned & has_seeds & not_store).astype(BOARD_DTYPE)

    def apply_fn(state, action):
        pit = action
        seeds = state.seed_counts[pit]
        sc = state.seed_counts.at[pit].set(0)
        start_pos = track_pos[pit]

        def sow_one(i, counts):
            pos = (start_pos + i + 1) % track_len
            cell = track_arr[pos]
            return counts.at[cell].add(1)

        sc = jax.lax.fori_loop(0, seeds, sow_one, sc)
        pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(pit)).at[num_players].set(ACTION_DTYPE(pit))
        return state._replace(seed_counts=sc, previous_actions=pa)

    return legal_fn, apply_fn, initial_seeds, pit_owner_init


def compile_dice_move(topology, piece_idx, num_players):
    """Compile dice-based track movement (backgammon-style)."""
    n = topology.num_sites

    def legal_fn(state):
        return (state.board == state.current_player).any(axis=0).astype(BOARD_DTYPE)

    def apply_fn(state, action):
        old_pos = action
        dice_sum = state.dice_values.sum()
        new_pos = (old_pos + dice_sum) % n
        board = state.board.at[piece_idx, old_pos].set(EMPTY)
        board = board.at[piece_idx, new_pos].set(state.current_player)
        pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(new_pos)).at[num_players].set(ACTION_DTYPE(new_pos))
        return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def combine_move_fns(legal_fns, apply_fns, num_sites):
    """Combine multiple movement types into single legal/apply pair.

    OR the legal masks, dispatch apply by checking which mask matched.
    """
    from ..runtime import state as state_mod

    if len(legal_fns) == 1:
        def combined_legal(state):
            return legal_fns[0](state).flatten().astype(BOARD_DTYPE)
        return combined_legal, apply_fns[0]

    def combined_legal(state):
        masks = jnp.stack([fn(state).flatten() for fn in legal_fns])
        return masks.any(axis=0).astype(BOARD_DTYPE)

    def combined_apply(state, action):
        masks = jnp.stack([fn(state).flatten() for fn in legal_fns])
        move_idx = jnp.argmax(masks[:, action])
        return jax.lax.switch(move_idx, apply_fns, state, action)

    return combined_legal, combined_apply
