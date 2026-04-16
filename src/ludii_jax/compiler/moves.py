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


def _update_actions(state, dst, num_players):
    """Update previous_actions with the destination cell. Used by all apply_fns."""
    cp = state.current_player.astype(jnp.int32)
    return state.previous_actions.at[cp].set(
        ACTION_DTYPE(dst)).at[num_players].set(ACTION_DTYPE(dst))


def _build_dir_masks(directions, max_nb):
    """Build direction mask(s) from directions spec.

    directions: None (all), list of ints (same for both players),
    or tuple of (p1_dirs, p2_dirs) for per-player directions.
    Returns (dir_mask, per_player) where per_player is True if
    dir_mask is a (2, max_nb) array indexed by player.
    """
    if directions is None:
        return jnp.ones(max_nb, dtype=jnp.bool_), False
    if isinstance(directions, tuple) and len(directions) == 2:
        p1_dirs, p2_dirs = directions
        masks = jnp.zeros((2, max_nb), dtype=jnp.bool_)
        for d in p1_dirs:
            if d < max_nb: masks = masks.at[0, d].set(True)
        for d in p2_dirs:
            if d < max_nb: masks = masks.at[1, d].set(True)
        return masks, True
    else:
        mask = jnp.zeros(max_nb, dtype=jnp.bool_)
        for d in directions:
            if d < max_nb: mask = mask.at[d].set(True)
        return mask, False


def compile_step(topology, slide_lookup, piece_idx, num_players, distance=1,
                 directions=None, reset_chain=False,
                 promote_from=-1, promote_to=-1, promo_rows=None,
                 to_empty=False):
    """Compile step movement: move exactly `distance` cells.

    directions: list, tuple of (p1_dirs, p2_dirs), or None.
    reset_chain: if True, reset extra_turn_fn_idx to -1 (ends any chain capture).
    promote_from/to: piece indices for promotion. promo_rows: (2, n) bool mask.
    to_empty: if True, only step to empty cells (no capture-by-step).
    """
    n = topology.num_sites
    max_nb = topology.max_neighbors
    adj = jnp.array(topology.adjacency)
    arange_n = jnp.arange(n, dtype=jnp.int32)

    dir_masks, per_player = _build_dir_masks(directions, max_nb)
    do_promote = promote_from >= 0 and promote_to >= 0 and promo_rows is not None and piece_idx == promote_from

    def legal_fn(state):
        owned = (state.board[piece_idx] == state.current_player)
        dests = slide_lookup[:, :, distance]  # (max_nb, n)
        on_board = dests < n
        if to_empty:
            dest_ok = ((state.board == EMPTY).astype(BOARD_DTYPE)).min(axis=0).astype(jnp.bool_)[dests.clip(0, n - 1)]
        else:
            friendly = (state.board == state.current_player).any(axis=0)
            dest_ok = ~friendly[dests.clip(0, n - 1)]
        valid = owned[jnp.newaxis, :] & dest_ok & on_board

        if per_player:
            dm = dir_masks[state.current_player.astype(jnp.int32)]
        else:
            dm = dir_masks
        valid = valid & dm[:, jnp.newaxis]

        if reset_chain:
            is_forced = (state.extra_turn_fn_idx >= 0)
            valid = jnp.where(is_forced, jnp.zeros_like(valid), valid)

        flat_idx = arange_n[jnp.newaxis, :] * n + dests.clip(0, n - 1)
        mask = jnp.zeros(n * n, dtype=BOARD_DTYPE)
        mask = mask.at[flat_idx.flatten()].set(valid.flatten().astype(BOARD_DTYPE))
        return mask

    def apply_fn(state, action):
        src, dst = action // n, action % n
        if not to_empty:
            board = state.board.at[:, dst].set(EMPTY)  # capture at destination
        else:
            board = state.board
        board = board.at[piece_idx, src].set(EMPTY)
        board = board.at[piece_idx, dst].set(state.current_player)
        # Promotion: if piece is on promotion row, swap to promoted piece
        if do_promote:
            should_promote = promo_rows[state.current_player.astype(jnp.int32), dst]
            board = jnp.where(should_promote,
                              board.at[promote_from, dst].set(EMPTY).at[promote_to, dst].set(state.current_player),
                              board)
        pa = _update_actions(state, dst, num_players)
        if reset_chain:
            return state._replace(board=board, previous_actions=pa,
                                  extra_turn_fn_idx=ACTION_DTYPE(-1))
        return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_hop(topology, slide_lookup, hop_between, piece_idx, num_players,
                hop_over="opponent", capture=True, directions=None,
                chain_capture=False,
                promote_from=-1, promote_to=-1, promo_rows=None):
    """Compile hop movement: jump over a piece to land beyond it.

    directions: list, tuple of (p1_dirs, p2_dirs), or None.
    chain_capture: if True, check for chain captures after each hop.
    promote_from/to: piece indices for promotion. promo_rows: (2, n) bool mask.
    """
    n = topology.num_sites
    max_nb = topology.max_neighbors
    arange_n = jnp.arange(n, dtype=jnp.int32)
    hop_over_friendly = (hop_over == "mover")
    hop_over_any = (hop_over == "any")
    do_promote = promote_from >= 0 and promote_to >= 0 and promo_rows is not None and piece_idx == promote_from

    dir_masks, per_player = _build_dir_masks(directions, max_nb)

    def _get_dir_mask(state):
        if per_player:
            return dir_masks[state.current_player.astype(jnp.int32)]
        return dir_masks

    def legal_fn(state):
        owned = (state.board[piece_idx] == state.current_player)
        dm = _get_dir_mask(state)
        if hop_over_any:
            hop_mask = (state.board != EMPTY).any(axis=0)
        elif hop_over_friendly:
            hop_mask = (state.board == state.current_player).any(axis=0)
        else:
            hop_mask = ((state.board != EMPTY) & (state.board != state.current_player)).any(axis=0)

        between = slide_lookup[:, :, 1]  # (max_nb, n)
        dests = slide_lookup[:, :, 2]    # (max_nb, n)
        on_board = dests < n
        has_hop_piece = hop_mask[between.clip(0, n - 1)]
        valid = owned[jnp.newaxis, :] & on_board & has_hop_piece
        valid = valid & dm[:, jnp.newaxis]

        if hop_over_any:
            # Land on empty only
            empty = ((state.board == EMPTY).astype(BOARD_DTYPE)).min(axis=0).astype(jnp.bool_)
            valid = valid & empty[dests.clip(0, n - 1)]
        elif hop_over_friendly:
            enemy = ((state.board != EMPTY) & (state.board != state.current_player)).any(axis=0)
            valid = valid & enemy[dests.clip(0, n - 1)]
        else:
            occupied = (state.board != EMPTY).any(axis=0)
            valid = valid & ~occupied[dests.clip(0, n - 1)]

        flat_idx = arange_n[jnp.newaxis, :] * n + dests.clip(0, n - 1)
        mask = jnp.zeros(n * n, dtype=BOARD_DTYPE)
        mask = mask.at[flat_idx.flatten()].set(valid.flatten().astype(BOARD_DTYPE))

        if chain_capture:
            forced_from = state.extra_turn_fn_idx
            is_forced = (forced_from >= 0)
            from_mask = (arange_n == forced_from)
            forced_valid = from_mask[jnp.newaxis, :] & on_board & has_hop_piece & dm[:, jnp.newaxis]
            if not hop_over_friendly:
                forced_valid = forced_valid & ~occupied[dests.clip(0, n - 1)]
            forced_flat = jnp.zeros(n * n, dtype=BOARD_DTYPE)
            forced_flat = forced_flat.at[flat_idx.flatten()].set(
                forced_valid.flatten().astype(BOARD_DTYPE))
            mask = jnp.where(is_forced, forced_flat, mask)

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

            # Promotion: if piece landed on promotion row, swap layers
            if do_promote:
                should_promote = promo_rows[state.current_player.astype(jnp.int32), dst]
                board = jnp.where(should_promote,
                                  board.at[promote_from, dst].set(EMPTY).at[promote_to, dst].set(state.current_player),
                                  board)

            pa = _update_actions(state, dst, num_players)

            if chain_capture:
                dm = _get_dir_mask(state)
                between_from_dst = slide_lookup[:, dst, 1]
                landing_from_dst = slide_lookup[:, dst, 2]
                on_board_c = (landing_from_dst < n) & (between_from_dst < n)
                enemy_mask = ((board != EMPTY) & (board != state.current_player)).any(axis=0)
                empty_mask = ((board == EMPTY).astype(BOARD_DTYPE)).min(axis=0).astype(jnp.bool_)
                can_chain = on_board_c & dm & \
                    enemy_mask[between_from_dst.clip(0, n - 1)] & \
                    empty_mask[landing_from_dst.clip(0, n - 1)]
                has_chain = can_chain.any()
                phase_adj = jax.lax.select(has_chain, BOARD_DTYPE(-1), BOARD_DTYPE(0))
                forced = jax.lax.select(has_chain, ACTION_DTYPE(dst), ACTION_DTYPE(-1))
                return state._replace(
                    board=board, previous_actions=pa,
                    phase_step_count=state.phase_step_count + phase_adj,
                    extra_turn_fn_idx=forced)
            return state._replace(board=board, previous_actions=pa)
    else:
        def apply_fn(state, action):
            src, dst = action // n, action % n
            board = state.board.at[piece_idx, dst].set(state.current_player)
            board = board.at[piece_idx, src].set(EMPTY)
            pa = _update_actions(state, dst, num_players)
            return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_slide(topology, slide_lookup, piece_idx, num_players, max_distance=None,
                   blocked_cells=None, directions=None):
    """Compile slide movement: move any number of cells in a direction until blocked.

    blocked_cells: optional jnp bool array (n,) — cells that block sliding even when empty.
    directions: list of direction indices to allow (None = all).
    """
    n = topology.num_sites
    max_nb = topology.max_neighbors
    if max_distance is None:
        max_distance = min(n, 32)
    arange_n = jnp.arange(n, dtype=ACTION_DTYPE)
    general_idx = jnp.indices((n, max_nb, max_distance), dtype=ACTION_DTYPE)[2]
    occupied_pad = jnp.ones((n, max_nb, 1), dtype=ACTION_DTYPE)
    _blocked = blocked_cells if blocked_cells is not None else jnp.zeros(n, dtype=jnp.bool_)

    # Direction mask for slide
    if directions is not None:
        _dir_mask = jnp.zeros(max_nb, dtype=jnp.bool_)
        for d in directions:
            if d < max_nb:
                _dir_mask = _dir_mask.at[d].set(True)
    else:
        _dir_mask = jnp.ones(max_nb, dtype=jnp.bool_)

    def legal_fn(state):
        occupied = ((state.board != EMPTY).any(axis=0) | _blocked).astype(BOARD_DTYPE)
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

        # Apply direction restriction: zero out columns for disabled directions
        if directions is not None:
            dir_valid = jnp.repeat(_dir_mask, max_distance)  # (max_nb * max_distance,)
            # Reshape valid_dests was (n, max_nb*max_distance), mask built from it
            # Simpler: recompute only for allowed directions
            allowed_dests = jnp.where(
                (_dir_mask[:, jnp.newaxis] & (general_idx < slide_until[:, :, jnp.newaxis])),
                slide_indices, n + 1
            ).reshape(n, -1)
            mask = jnp.zeros((n, n), dtype=BOARD_DTYPE)
            mask = mask.at[arange_n[:, jnp.newaxis], allowed_dests].set(1)
            mask = mask.at[arange_n, arange_n].set(0)

        piece_mask = (state.board[piece_idx] == state.current_player).astype(BOARD_DTYPE)
        mask = jnp.where(piece_mask[:, jnp.newaxis], mask, jnp.zeros_like(mask))
        return mask.flatten()

    def apply_fn(state, action):
        src, dst = action // n, action % n
        board = state.board.at[piece_idx, dst].set(state.current_player)
        board = board.at[piece_idx, src].set(EMPTY)
        pa = _update_actions(state, dst, num_players)
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
            pa = _update_actions(state, dst, num_players)
            return state._replace(board=board, previous_actions=pa)
    else:
        def apply_fn(state, action):
            src, dst = action // n, action % n
            board = state.board.at[piece_idx, dst].set(state.current_player)
            board = board.at[piece_idx, src].set(EMPTY)
            pa = _update_actions(state, dst, num_players)
            return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_place(topology, piece_idx, num_players):
    """Compile placement: place a piece on any empty site."""
    n = topology.num_sites

    def legal_fn(state):
        return ((state.board == EMPTY).astype(BOARD_DTYPE)).min(axis=0).astype(jnp.bool_).astype(BOARD_DTYPE)

    def apply_fn(state, action):
        board = state.board.at[piece_idx, action].set(state.current_player)
        pa = _update_actions(state, action, num_players)
        return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_stack_place(topology, max_height, num_players):
    """Compile stacking placement: place piece on top of stack at any site.

    Board shape: (max_height, num_sites). Layer 0 = bottom of stack.
    Legal: any site where top of stack < max_height.
    Apply: place at lowest empty layer.
    """
    n = topology.num_sites

    def legal_fn(state):
        # Legal if at least one layer is empty (stack not full)
        has_space = (state.board == EMPTY).any(axis=0).astype(BOARD_DTYPE)
        return has_space

    def apply_fn(state, action):
        cell = action
        # Find lowest empty layer at this cell
        layers = state.board[:, cell]
        is_empty = (layers == EMPTY).astype(jnp.int32)
        # First empty layer = argmax of (is_empty) since argmax returns first True
        layer = jnp.argmax(is_empty).astype(jnp.int32)
        board = state.board.at[layer, cell].set(state.current_player)
        pa = _update_actions(state, action, num_players)
        return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def compile_sow(topology, num_players, initial_seeds=4, has_stores=True,
                stores_in_track=True):
    """Compile mancala sowing: pick a pit, distribute seeds along track.

    Ludii mancalaBoard 2 N layout (14 cells for N=6):
      Cell 0: left store, Cells 1-N: P1 pits (bottom),
      Cells N+1..2N: P2 pits (top), Cell 2N+1: right store.

    When stores_in_track=True (Kalah): each player skips opponent's store.
    Per-player tracks: P1 skips cell 0, P2 skips cell n-1.
    Kalah capture: if last seed lands in empty own pit, capture opposite.
    """
    n = topology.num_sites
    if has_stores and n >= 4:
        pits_per_player = (n - 2) // 2
        left_store = 0
        right_store = n - 1
        is_store = jnp.zeros(n, dtype=jnp.bool_).at[left_store].set(True).at[right_store].set(True)
    else:
        pits_per_player = n // 2
        left_store = -1
        right_store = -1
        is_store = jnp.zeros(n, dtype=jnp.bool_)

    ppp = pits_per_player if has_stores and n >= 4 else n // 2

    # Build tracks
    if has_stores and n >= 4 and stores_in_track:
        # Full track with both stores
        full_track = list(range(1, ppp + 1)) + [n - 1] + list(range(2 * ppp, ppp, -1)) + [0]
        # Per-player tracks: skip opponent's store
        # P1 (player 0): skip cell 0 (left store = P2's)
        p1_track = [c for c in full_track if c != 0]   # 13 cells
        # P2 (player 1): skip cell n-1 (right store = P1's)
        p2_track = [c for c in full_track if c != n - 1]  # 13 cells
        # Pad to same length, use cell 0 as padding (won't be reached)
        max_tlen = max(len(p1_track), len(p2_track))
        p1_track = p1_track + [0] * (max_tlen - len(p1_track))
        p2_track = p2_track + [0] * (max_tlen - len(p2_track))
        tracks = jnp.array([p1_track, p2_track], dtype=ACTION_DTYPE)  # (2, tlen)
        track_len = jnp.array([len([c for c in full_track if c != 0]),
                               len([c for c in full_track if c != n - 1])], dtype=ACTION_DTYPE)
        # Build track_pos per player
        track_pos = jnp.full((2, n), 0, dtype=ACTION_DTYPE)
        for pi, trk in enumerate([p1_track, p2_track]):
            for i, cell in enumerate(trk[:int(track_len[pi])]):
                track_pos = track_pos.at[pi, cell].set(ACTION_DTYPE(i))
    elif has_stores and n >= 4:
        # No stores in track (Oware)
        track = list(range(1, ppp + 1)) + list(range(2 * ppp, ppp, -1))
        tlen = len(track)
        # Same track for both players
        tracks = jnp.array([track, track], dtype=ACTION_DTYPE)
        track_len = jnp.array([tlen, tlen], dtype=ACTION_DTYPE)
        track_pos = jnp.full((2, n), 0, dtype=ACTION_DTYPE)
        for i, cell in enumerate(track):
            track_pos = track_pos.at[0, cell].set(ACTION_DTYPE(i))
            track_pos = track_pos.at[1, cell].set(ACTION_DTYPE(i))
    else:
        half = n // 2
        track = list(range(half)) + list(range(n - 1, half - 1, -1))
        tlen = len(track)
        tracks = jnp.array([track, track], dtype=ACTION_DTYPE)
        track_len = jnp.array([tlen, tlen], dtype=ACTION_DTYPE)
        track_pos = jnp.full((2, n), 0, dtype=ACTION_DTYPE)
        for i, cell in enumerate(track):
            track_pos = track_pos.at[0, cell].set(ACTION_DTYPE(i))
            track_pos = track_pos.at[1, cell].set(ACTION_DTYPE(i))

    # Opposite pit lookup for Kalah capture: opposite(i) = i+ppp for P1 pits, i-ppp for P2 pits
    opposite = jnp.full(n, n, dtype=ACTION_DTYPE)  # n = no opposite
    if has_stores and n >= 4:
        for i in range(1, ppp + 1):
            opposite = opposite.at[i].set(ACTION_DTYPE(i + ppp))
        for i in range(ppp + 1, 2 * ppp + 1):
            opposite = opposite.at[i].set(ACTION_DTYPE(i - ppp))

    # Store indices per player: P1's store = right (n-1), P2's store = left (0)
    player_store = jnp.array([n - 1, 0], dtype=ACTION_DTYPE) if has_stores and n >= 4 else jnp.array([0, 0], dtype=ACTION_DTYPE)

    # Pit ownership
    if has_stores and n >= 4:
        pit_owner_init = jnp.full(n, BOARD_DTYPE(-1))
        for i in range(1, ppp + 1):
            pit_owner_init = pit_owner_init.at[i].set(BOARD_DTYPE(0))
        for i in range(ppp + 1, 2 * ppp + 1):
            pit_owner_init = pit_owner_init.at[i].set(BOARD_DTYPE(1))
        pit_owner_init = pit_owner_init.at[right_store].set(BOARD_DTYPE(0))
        pit_owner_init = pit_owner_init.at[left_store].set(BOARD_DTYPE(1))
    else:
        half = n // 2
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
        cp = state.current_player.astype(jnp.int32)
        seeds = state.seed_counts[pit]
        sc = state.seed_counts.at[pit].set(0)
        start_pos = track_pos[cp, pit]
        tlen = track_len[cp]
        trk = tracks[cp]

        def sow_one(i, counts):
            pos = (start_pos + i + 1) % tlen
            cell = trk[pos]
            return counts.at[cell].add(1)

        sc = jax.lax.fori_loop(0, seeds, sow_one, sc)

        # Find where last seed landed
        last_pos = (start_pos + seeds) % tlen
        last_cell = trk[last_pos]

        # Extra turn if last seed landed in player's store
        landed_in_own_store = is_store[last_cell] & (state.pit_owner[last_cell] == cp)

        # Kalah capture: if last seed lands in own empty pit (now has 1) and opposite has seeds
        if stores_in_track:
            is_own_pit = (state.pit_owner[last_cell] == cp) & ~is_store[last_cell]
            was_empty = (sc[last_cell] == 1)  # exactly 1 = was empty before sowing this seed
            opp = opposite[last_cell]
            opp_has_seeds = (opp < n) & (sc[opp.clip(0, n - 1)] > 0)
            do_capture = is_own_pit & was_empty & opp_has_seeds & ~landed_in_own_store

            my_store = player_store[cp]
            captured = sc[opp.clip(0, n - 1)] + sc[last_cell]
            sc = jax.lax.select(do_capture, sc.at[my_store].add(captured).at[last_cell].set(0).at[opp.clip(0, n - 1)].set(0), sc)

        phase_adj = jax.lax.select(landed_in_own_store, BOARD_DTYPE(-1), BOARD_DTYPE(0))

        pa = state.previous_actions.at[cp].set(ACTION_DTYPE(pit)).at[num_players].set(ACTION_DTYPE(last_cell))
        return state._replace(seed_counts=sc, previous_actions=pa,
                              phase_step_count=state.phase_step_count + phase_adj)

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
        pa = _update_actions(state, new_pos, num_players)
        return state._replace(board=board, previous_actions=pa)

    return legal_fn, apply_fn


def combine_move_fns(legal_fns, apply_fns, num_sites, priority_indices=None):
    """Combine multiple movement types into single legal/apply pair.

    OR the legal masks, dispatch apply by checking which mask matched.
    priority_indices: if set, these legal_fn indices have priority (forced captures).
    When any priority move is legal, non-priority moves are masked out.
    """
    from ..runtime import state as state_mod

    if len(legal_fns) == 1:
        def combined_legal(state):
            return legal_fns[0](state).flatten().astype(BOARD_DTYPE)
        return combined_legal, apply_fns[0]

    if priority_indices is not None:
        _pri_mask = jnp.zeros(len(legal_fns), dtype=jnp.bool_)
        for pi in priority_indices:
            _pri_mask = _pri_mask.at[pi].set(True)

    def combined_legal(state):
        masks = jnp.stack([fn(state).flatten() for fn in legal_fns])
        if priority_indices is not None:
            # Weight priority moves: if any exists, mask out non-priority
            pri_only = masks * _pri_mask[:, jnp.newaxis]
            has_priority = pri_only.any()
            all_moves = masks.any(axis=0).astype(BOARD_DTYPE)
            pri_flat = pri_only.any(axis=0).astype(BOARD_DTYPE)
            return jnp.where(has_priority, pri_flat, all_moves)
        return masks.any(axis=0).astype(BOARD_DTYPE)

    def combined_apply(state, action):
        masks = jnp.stack([fn(state).flatten() for fn in legal_fns])
        move_idx = jnp.argmax(masks[:, action])
        return jax.lax.switch(move_idx, apply_fns, state, action)

    return combined_legal, combined_apply
