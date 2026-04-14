"""
Top-level compiler: .lud file → JAX Environment.

    from ludii_jax import compile
    env = compile("games/chess.lud")
    state = env.init(jax.random.PRNGKey(0))
"""

import re

from .parser.parse import parse, find_child, find_all, get_text
from .analysis.game_info import extract_game_info
from .analysis.topology import BoardTopology
from .runtime.lookup import (
    build_slide_lookup, build_hop_between_lookup,
    build_line_indices, build_edge_mask,
)
from .runtime.state import build_game_state_class, BOARD_DTYPE, EMPTY
from .runtime.environment import Environment
from .compiler.moves import (
    compile_step, compile_hop, compile_slide, compile_leap,
    compile_place, compile_sow, compile_dice_move,
    combine_move_fns,
)
from .compiler.effects import (
    compile_custodial_capture, compile_flip, compile_set_score,
    compile_extra_turn, chain_effects,
)
from .compiler.conditions import (
    compile_line_win, compile_line_loss, compile_no_moves_loss,
    compile_captured_all, compile_full_board_draw,
    compile_full_board_by_score, combine_end_conditions,
)
from .compiler.compose import compose_game, make_alternating_player_fn

import jax.numpy as jnp


def compile(lud_text_or_path: str):
    """Compile a Ludii .lud game to a JAX Environment.

    Args:
        lud_text_or_path: Either raw .lud text or a file path.

    Returns:
        Environment ready for init/step/observe.
    """
    # Load file if path
    if lud_text_or_path.endswith('.lud'):
        with open(lud_text_or_path) as f:
            lud_text = f.read()
    else:
        lud_text = lud_text_or_path

    # Parse
    tree = parse(lud_text)

    # Analyze
    info = extract_game_info(tree)
    topo = info.topology
    np = info.num_players

    # Build lookup tables
    slide_lookup = build_slide_lookup(topo)
    hop_between = build_hop_between_lookup(topo, slide_lookup)

    # Build game state class
    GameState, defaults = build_game_state_class(info)

    # Determine action size and compile movement
    piece_idx = 0  # default piece
    piece_names = [p.name for p in info.pieces] if info.pieces else ["token"]

    if info.is_mancala:
        legal_fn, apply_fn, initial_seeds, pit_owner = compile_sow(topo, np, info.mancala_seeds)
        action_size = topo.num_sites

        def start_fn(state):
            sc = jnp.full(topo.num_sites, BOARD_DTYPE(initial_seeds))
            return state._replace(seed_counts=sc, pit_owner=pit_owner)

    elif info.is_dice:
        legal_fn, apply_fn = compile_dice_move(topo, piece_idx, np)
        action_size = topo.num_sites
        start_fn = _build_start_fn(tree, info, topo)
        # If no pieces placed, auto-place on first/last few cells
        if "place" not in info.start_text and topo.num_sites >= 4:
            half = topo.num_sites // 2
            def dice_start(state):
                board = state.board
                # Place some pieces for each player
                for i in range(min(2, half)):
                    board = board.at[piece_idx, i].set(BOARD_DTYPE(0))
                    board = board.at[piece_idx, topo.num_sites - 1 - i].set(BOARD_DTYPE(1))
                return state._replace(board=board)
            start_fn = dice_start

    else:
        # Determine movement types and build combined legal/apply
        legal_fns = []
        apply_fns = []

        for pi, p in enumerate(info.pieces if info.pieces else [type('P', (), {'name': 'token'})()]):
            if info.has_step or (not info.has_hop and not info.has_slide and not info.has_leap):
                l, a = compile_step(topo, slide_lookup, pi, np)
                legal_fns.append(l)
                apply_fns.append(a)
            if info.has_hop:
                hop_over = "mover" if ("is Friend" in info.full_text and "between" in info.full_text) else "opponent"
                l, a = compile_hop(topo, slide_lookup, hop_between, pi, np, hop_over=hop_over)
                legal_fns.append(l)
                apply_fns.append(a)

        if info.has_slide:
            l, a = compile_slide(topo, slide_lookup, piece_idx, np)
            legal_fns.append(l)
            apply_fns.append(a)

        if info.has_leap:
            offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            l, a = compile_leap(topo, piece_idx, np, offsets)
            legal_fns.append(l)
            apply_fns.append(a)

        # If we have movement, use FROM_TO action space
        if legal_fns:
            legal_fn, apply_fn = combine_move_fns(legal_fns, apply_fns, topo.num_sites)
            action_size = topo.num_sites * topo.num_sites
        else:
            # Pure placement
            legal_fn, apply_fn = compile_place(topo, piece_idx, np)
            action_size = topo.num_sites

        # Check if the PLAY section specifically uses placement vs movement
        full_text = info.full_text
        # Get just the play section text (not piece definitions)
        play_section = full_text[full_text.find("play"):] if "play" in full_text else full_text
        play_has_add = any(kw in play_section for kw in ["move Add", "move Claim", "satisfy", "handSite"])
        play_has_movement = any(kw in play_section for kw in ["forEach Piece", "move Step", "move Hop", "move Slide", "move Leap"])
        has_placement = play_has_add or "handSite" in full_text or "Hand" in full_text
        has_movement = play_has_movement

        start_fn = _build_start_fn(tree, info, topo)

        # Route to placement if:
        # 1. Play section uses Add/Claim/satisfy/handSite (pure placement)
        # 2. Play section has both placement and movement but no start pieces
        if has_placement and not has_movement:
            legal_fn, apply_fn = compile_place(topo, piece_idx, np)
            action_size = topo.num_sites
        elif has_placement and has_movement and "place" not in info.start_text:
            legal_fn, apply_fn = compile_place(topo, piece_idx, np)
            action_size = topo.num_sites

    # Compile effects
    effects = []
    if info.has_capture:
        adj_lookup = None  # could build from topology
        effects.append(compile_custodial_capture(topo, adj_lookup, piece_idx, num_players=np))
    if info.has_score:
        effects.append(compile_set_score(np))
    effects_fn = chain_effects(effects)

    # Compile end conditions
    end_fns = []
    full_text = info.full_text

    if "is Line" in full_text:
        for m in re.finditer(r'is Line (\d+).*?result (\w+) (\w+)', full_text):
            n_line = int(m.group(1))
            outcome = m.group(3)
            line_idx = build_line_indices(topo, n_line)
            if outcome == "Win":
                end_fns.append(compile_line_win(line_idx, piece_idx, np))
            elif outcome == "Loss":
                end_fns.append(compile_line_loss(line_idx, piece_idx, np))

    if "no Moves" in full_text or "no_legal" in full_text:
        end_fns.append(compile_no_moves_loss(np))

    if "no Pieces" in full_text:
        end_fns.append(compile_captured_all(np))

    if "is Full" in full_text:
        if "by_score" in full_text.lower() or "by Score" in full_text:
            end_fns.append(compile_full_board_by_score(np))
        else:
            end_fns.append(compile_full_board_draw(np))

    if not end_fns:
        end_fns.append(compile_no_moves_loss(np))

    end_fn = combine_end_conditions(end_fns, np)

    # Compose
    next_player_fn = make_alternating_player_fn(np)

    game_rules = compose_game(
        action_size=action_size,
        legal_fn=legal_fn,
        apply_fn=apply_fn,
        effects_fn=effects_fn,
        end_fn=end_fn,
        next_player_fn=next_player_fn,
        start_fn=start_fn,
        num_players=np,
    )

    return Environment(game_rules, GameState, info)


def _build_start_fn(tree, info, topo):
    """Build the start function that places initial pieces."""
    rules = find_child(tree, "rules")
    if not rules:
        return lambda state: state

    full_text = get_text(rules)
    n = topo.num_sites
    piece_names = [p.name for p in info.pieces]
    piece_name = piece_names[0] if piece_names else "token"

    def resolve_name(raw):
        name = re.sub(r'\d+$', '', raw).lower()
        if not name:
            name = raw.lower()
        name = info.piece_name_map.get(name, name)
        if name not in piece_names:
            name = piece_name
        return name

    placements = []  # [(piece_idx, player, [cell_indices])]

    # Region-based placement: (sites Bottom), (sites Top), etc.
    for m in re.finditer(r'place\s+"([^"]+)"\s+(?:\(?sites\s+)(Bottom|Top|Left|Right)', full_text):
        pname_raw = m.group(1)
        region = m.group(2).lower()
        pname = resolve_name(pname_raw)
        player = 0 if pname_raw.endswith("1") else 1 if pname_raw.endswith("2") else 0
        if region in topo.regions:
            indices = [i for i in range(n) if topo.regions[region][i]]
        else:
            # Approximate: bottom = first quarter, top = last quarter
            quarter = max(n // 4, 1)
            if region == "bottom":
                indices = list(range(quarter))
            elif region == "top":
                indices = list(range(n - quarter, n))
            elif region == "left":
                indices = [i for i in range(n) if topo.site_coords[i][0] < topo.site_coords[n//2][0]]
            elif region == "right":
                indices = [i for i in range(n) if topo.site_coords[i][0] > topo.site_coords[n//2][0]]
            else:
                indices = []
        if pname in piece_names and indices:
            placements.append((piece_names.index(pname), player, indices))

    # Expand pattern: (expand (sites Bottom/Top))
    if not placements:
        for m in re.finditer(r'place\s+"([^"]+)"\s+expand\s+sites\s+(Bottom|Top)', full_text):
            pname_raw = m.group(1)
            region = m.group(2).lower()
            pname = resolve_name(pname_raw)
            player = 0 if pname_raw.endswith("1") else 1 if pname_raw.endswith("2") else 0
            if region in topo.regions:
                indices = [i for i in range(n) if topo.regions[region][i]]
            else:
                half = max(n // 4, 1)
                indices = list(range(half)) if region == "bottom" else list(range(n - half, n))
            if pname in piece_names and indices:
                placements.append((piece_names.index(pname), player, indices))

    # Direct cell index: place "Name" N or place "Name" N count:M
    for m in re.finditer(r'place\s+(?:Stack\s+)?"([^"]+)"\s+(\d+)', full_text):
        pname_raw = m.group(1)
        cell = int(m.group(2))
        if cell < n:
            pname = resolve_name(pname_raw)
            player = 0 if pname_raw.endswith("1") else 1 if pname_raw.endswith("2") else 0
            if pname in piece_names:
                placements.append((piece_names.index(pname), player, [cell]))

    # Coord placement: place "Name" "A1" or place "Name" coord:"A1"
    for m in re.finditer(r'place\s+"([^"]+)"\s+(?:coord:)?"([A-Za-z]\d+)"', full_text):
        pname_raw = m.group(1)
        coord = m.group(2)
        pname = resolve_name(pname_raw)
        player = 0 if pname_raw.endswith("1") else 1 if pname_raw.endswith("2") else 0
        # Convert chess notation to index
        col = ord(coord[0].upper()) - ord('A')
        row = int(coord[1:]) - 1
        # Find nearest site in topology
        if topo.site_coords:
            best_idx = min(range(n), key=lambda i: abs(topo.site_coords[i][0] - col) + abs(topo.site_coords[i][1] - row))
            if pname in piece_names:
                placements.append((piece_names.index(pname), player, [best_idx]))

    # Site-list pattern: place "Name" sites 2 3 4 ...
    for m in re.finditer(r'place\s+"([^"]+)"\s+(?:\(?sites\s+)?\{?(\d[\d\s]+)\}?', full_text):
        pname_raw = m.group(1)
        indices = [int(x) for x in re.findall(r'\d+', m.group(2))]
        indices = [i for i in indices if i < n]
        pname = resolve_name(pname_raw)
        player = 0 if pname_raw.endswith("1") else 1 if pname_raw.endswith("2") else 0
        if pname in piece_names and indices:
            placements.append((piece_names.index(pname), player, indices))

    # Row pattern: place "Name" Row N
    for m in re.finditer(r'place\s+"([^"]+)"\s+(?:\(?sites )?Row (\d+)\)?', full_text):
        pname_raw = m.group(1)
        row = int(m.group(2))
        pname = resolve_name(pname_raw)
        player = 0 if pname_raw.endswith("1") else 1 if pname_raw.endswith("2") else 0
        # Compute row indices from topology coords
        if topo.site_coords:
            row_indices = [i for i, (x, y) in enumerate(topo.site_coords) if int(round(y)) == row]
            if pname in piece_names and row_indices:
                placements.append((piece_names.index(pname), player, row_indices))

    if not placements:
        # Fallback: auto-place for movement games
        has_movement = info.has_step or info.has_hop or info.has_slide
        has_place_in_start = "place" in full_text.split("play")[0] if "play" in full_text else False
        if has_movement and has_place_in_start and n >= 4:
            # Place on first/last rows
            first_row = [i for i, (x, y) in enumerate(topo.site_coords) if y == min(yy for _, yy in topo.site_coords)]
            last_row = [i for i, (x, y) in enumerate(topo.site_coords) if y == max(yy for _, yy in topo.site_coords)]
            if first_row and last_row:
                placements.append((0, 0, first_row))
                placements.append((0, 1, last_row))

    if not placements:
        return lambda state: state

    def start_fn(state):
        board = state.board
        for piece_idx, player, indices in placements:
            for idx in indices:
                if idx < n:
                    board = board.at[piece_idx, idx].set(BOARD_DTYPE(player))
        return state._replace(board=board)

    return start_fn
