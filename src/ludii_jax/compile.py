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
from .analysis.sites import evaluate_sites
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
        # Detect stores-in-track: Kalah-style has "moveAgain" tied to store landing
        # Oware-style has no store sowing (track starts at pit 1, not store 0)
        stores_in_track = "moveAgain" in info.full_text and "mapEntry" in info.full_text
        legal_fn, apply_fn, initial_seeds, pit_owner = compile_sow(
            topo, np, info.mancala_seeds, stores_in_track=stores_in_track)
        action_size = topo.num_sites

        # Initial seeds: only on pits, not on stores
        ppp = (topo.num_sites - 2) // 2 if topo.num_sites >= 4 else topo.num_sites // 2
        def start_fn(state, _seeds=initial_seeds, _po=pit_owner, _n=topo.num_sites, _ppp=ppp):
            sc = jnp.zeros(_n, dtype=BOARD_DTYPE)
            # Set seeds only on pit cells (1..ppp and ppp+1..2*ppp)
            for i in range(1, _ppp + 1):
                sc = sc.at[i].set(BOARD_DTYPE(_seeds))
            for i in range(_ppp + 1, 2 * _ppp + 1):
                sc = sc.at[i].set(BOARD_DTYPE(_seeds))
            return state._replace(seed_counts=sc, pit_owner=_po)

    elif info.is_dice:
        legal_fn, apply_fn = compile_dice_move(topo, piece_idx, np)
        action_size = topo.num_sites
        base_start = _build_start_fn(tree, info, topo)

        # Dice games: always place pieces on the board
        n_sites = topo.num_sites
        # Use first/last cells (avoid bottom/top which overlap on 1-row boards)
        num_start = min(3, n_sites // 3) or 1
        p1_cells = list(range(num_start))
        p2_cells = list(range(n_sites - num_start, n_sites))

        def dice_start(state, _base=base_start, _p1=p1_cells, _p2=p2_cells, _pi=piece_idx):
            state = _base(state)
            board = state.board
            for c in _p1:
                board = board.at[_pi, c].set(BOARD_DTYPE(0))
            for c in _p2:
                board = board.at[_pi, c].set(BOARD_DTYPE(1))
            return state._replace(board=board)
        start_fn = dice_start

    else:
        # ============================================================
        # Structural mechanic detection from the play section's parse tree
        # ============================================================
        rules_node = find_child(tree, "rules")
        play_text = ""
        if rules_node:
            rules_content = find_child(rules_node, "rules_content")
            if rules_content:
                for item in find_all(rules_content, "rules_item"):
                    play_node = find_child(item, "play")
                    if play_node:
                        play_text = get_text(play_node)
                        break

        # Classify the PRIMARY mechanic from the play section structure
        if "move Add" in play_text or "move Claim" in play_text:
            mechanic = "PLACE"
        elif "satisfy" in play_text:
            mechanic = "PLACE"  # puzzle = placement
        elif "move Remove" in play_text and "forEach Piece" not in play_text:
            mechanic = "REMOVE"
        elif "move Select" in play_text and "forEach Piece" not in play_text and "move Step" not in play_text:
            mechanic = "SELECT"
        elif "forEach Site" in play_text and "forEach Piece" not in play_text:
            mechanic = "PLACE"
        elif "handSite" in play_text and "forEach Piece" not in play_text:
            mechanic = "PLACE"
        elif "forEach Piece" in play_text:
            mechanic = "FOREACH_PIECE"
        elif "move Step" in play_text or "move Hop" in play_text or "move Slide" in play_text:
            mechanic = "MOVEMENT"
        else:
            mechanic = "PLACE"  # default: placement

        start_fn = _build_start_fn(tree, info, topo)

        # ============================================================
        # Compile based on structural mechanic
        # ============================================================
        if mechanic == "PLACE":
            legal_fn, apply_fn = compile_place(topo, piece_idx, np)
            action_size = topo.num_sites

        elif mechanic == "REMOVE":
            def select_legal(state):
                return (state.board != EMPTY).any(axis=0).astype(BOARD_DTYPE)
            def select_apply(state, action):
                board = state.board.at[:, action].set(EMPTY)
                pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(action)).at[np].set(ACTION_DTYPE(action))
                return state._replace(board=board, previous_actions=pa)
            legal_fn, apply_fn = select_legal, select_apply
            action_size = topo.num_sites

        elif mechanic == "SELECT":
            def select_legal(state):
                return (state.board == state.current_player).any(axis=0).astype(BOARD_DTYPE)
            def select_apply(state, action):
                board = state.board.at[:, action].set(EMPTY)
                pa = state.previous_actions.at[state.current_player].set(ACTION_DTYPE(action)).at[np].set(ACTION_DTYPE(action))
                return state._replace(board=board, previous_actions=pa)
            legal_fn, apply_fn = select_legal, select_apply
            action_size = topo.num_sites

        else:
            # FOREACH_PIECE or MOVEMENT: compile step/hop/slide/leap
            legal_fns = []
            apply_fns = []

            # Detect direction restrictions from piece definitions
            piece_text = play_text
            for p in info.pieces:
                piece_text += " " + info.piece_content.get(p.name, "")
            # Direction indices for 8-dir grid (Ludii convention, row 0 = bottom):
            # 0=N(up), 1=NE, 2=E(right), 3=SE, 4=S(down), 5=SW, 6=W(left), 7=NW
            step_dirs = None
            hop_dirs = None
            if topo.max_neighbors == 8:
                DIAG = [1, 3, 5, 7]
                ORTHO = [0, 2, 4, 6]
                # Per-player forward directions
                P1_FWD = [0, 1, 7]       # N, NE, NW
                P1_FWD_DIAG = [1, 7]     # NE, NW
                P2_FWD = [4, 3, 5]       # S, SE, SW
                P2_FWD_DIAG = [3, 5]     # SE, SW

                if "Diagonal" in piece_text and "Orthogonal" not in piece_text:
                    step_dirs = DIAG
                    hop_dirs = DIAG
                elif "Orthogonal" in piece_text and "Diagonal" not in piece_text:
                    step_dirs = ORTHO
                    hop_dirs = ORTHO

                has_forward = "Forward" in piece_text or (info.has_set_forward and ("FR" in piece_text or "FL" in piece_text))
                if has_forward and info.has_set_forward:
                    # Per-player direction: (p1_dirs, p2_dirs) tuple
                    if step_dirs and set(step_dirs) == set(DIAG):
                        step_dirs = (P1_FWD_DIAG, P2_FWD_DIAG)
                    elif step_dirs:
                        step_dirs = ([d for d in step_dirs if d in P1_FWD],
                                     [d for d in step_dirs if d in P2_FWD])
                    else:
                        step_dirs = (P1_FWD, P2_FWD)
                    if hop_dirs and set(hop_dirs) == set(DIAG):
                        hop_dirs = (P1_FWD_DIAG, P2_FWD_DIAG)
                    else:
                        hop_dirs = (P1_FWD_DIAG, P2_FWD_DIAG)
                elif has_forward:
                    # Non-set_forward game: use P1 directions for all
                    if step_dirs and set(step_dirs) == set(DIAG):
                        step_dirs = P1_FWD_DIAG
                    elif step_dirs:
                        step_dirs = [d for d in step_dirs if d in P1_FWD]
                    else:
                        step_dirs = P1_FWD

            has_chain = info.has_extra_turn and "moveAgain" in info.full_text and info.has_hop
            has_priority = "priority" in play_text and info.has_hop
            hop_fn_indices = []  # track which indices in legal_fns are hop functions

            # Per-piece direction determination for promotion games
            # Regular pieces: forward-only; promoted pieces: all directions
            promote_from = -1  # piece idx that can promote
            promote_to = -1    # piece idx it promotes to
            if info.has_promote and len(info.pieces) >= 2:
                for pi, p in enumerate(info.pieces):
                    if "double" in p.name or "king" in p.name:
                        promote_to = pi
                    elif promote_from < 0:
                        promote_from = pi
                if promote_from < 0:
                    promote_from = 0
                if promote_to < 0:
                    promote_to = len(info.pieces) - 1

            # Build promotion row mask: P1 promotes at Top, P2 at Bottom
            promo_rows = None
            if promote_from >= 0 and promote_to >= 0:
                n_sites = topo.num_sites
                p1_promo = jnp.zeros(n_sites, dtype=jnp.bool_)
                p2_promo = jnp.zeros(n_sites, dtype=jnp.bool_)
                if "top" in topo.regions:
                    for i in range(n_sites):
                        if topo.regions["top"][i]:
                            p1_promo = p1_promo.at[i].set(True)
                if "bottom" in topo.regions:
                    for i in range(n_sites):
                        if topo.regions["bottom"][i]:
                            p2_promo = p2_promo.at[i].set(True)
                promo_rows = jnp.stack([p1_promo, p2_promo])  # (2, n)

            for pi, p in enumerate(info.pieces if info.pieces else [type('P', (), {'name': 'token'})()]):
                # Determine piece-specific directions
                if promote_from >= 0 and pi == promote_to:
                    # Promoted piece: all diagonal (or all directions)
                    pi_step_dirs = step_dirs if not isinstance(step_dirs, tuple) else None
                    pi_hop_dirs = hop_dirs if not isinstance(hop_dirs, tuple) else None
                    # If base dirs were diagonal, keep all diagonal for promoted piece
                    if topo.max_neighbors == 8:
                        pi_step_dirs = [1, 3, 5, 7]  # all diagonal
                        pi_hop_dirs = [1, 3, 5, 7]
                else:
                    pi_step_dirs = step_dirs
                    pi_hop_dirs = hop_dirs

                # Step to empty only when game has separate hop captures
                step_to_empty = has_priority and info.has_hop
                if info.has_step or (not info.has_hop and not info.has_slide and not info.has_leap):
                    l, a = compile_step(topo, slide_lookup, pi, np, directions=pi_step_dirs,
                                        reset_chain=has_chain,
                                        promote_from=promote_from, promote_to=promote_to,
                                        promo_rows=promo_rows, to_empty=step_to_empty)
                    legal_fns.append(l)
                    apply_fns.append(a)
                if info.has_hop:
                    hop_over = "mover" if ("is Friend" in info.full_text and "between" in info.full_text) else "opponent"
                    hop_fn_indices.append(len(legal_fns))
                    l, a = compile_hop(topo, slide_lookup, hop_between, pi, np, hop_over=hop_over,
                                       directions=pi_hop_dirs, chain_capture=has_chain,
                                       promote_from=promote_from, promote_to=promote_to,
                                       promo_rows=promo_rows)
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

            if legal_fns:
                pri = hop_fn_indices if has_priority and hop_fn_indices else None
                legal_fn, apply_fn = combine_move_fns(legal_fns, apply_fns, topo.num_sites, priority_indices=pri)
                action_size = topo.num_sites * topo.num_sites
            else:
                legal_fn, apply_fn = compile_place(topo, piece_idx, np)
                action_size = topo.num_sites

    # Compile effects
    effects = []
    # Custodial capture: compile when explicitly present in game text
    if "custodial" in info.full_text.lower():
        adj_lookup = None
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
    """Build the start function that places initial pieces.

    Uses the recursive site set evaluator from analysis.sites to handle
    compound expressions like difference(expand(sites Bottom, steps:2), sites Phase 1).
    """
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

    # Extract start section text from parse tree (not full rules text)
    start_text = ""
    rules = find_child(tree, "rules")
    if rules:
        rules_content = find_child(rules, "rules_content")
        if rules_content:
            for item in find_all(rules_content, "rules_item"):
                start_node = find_child(item, "start")
                if start_node:
                    start_text = get_text(start_node)
                    break

    if not start_text:
        # Fallback: use full rules text
        start_text = get_text(rules) if rules else ""

    placements = []  # [(piece_idx, player, [cell_indices])]

    # Find all place "Name" <expression> patterns
    # Split on 'place' boundaries to get each placement's expression
    place_parts = re.split(r'(?=place\s+(?:Stack\s+)?")', start_text)
    for part in place_parts:
        m = re.match(r'place\s+(?:Stack\s+)?"([^"]+)"\s+(.*)', part, re.DOTALL)
        if not m:
            continue
        pname_raw = m.group(1)
        expr = m.group(2).strip()

        pname = resolve_name(pname_raw)
        player = 0 if pname_raw.endswith("1") else 1 if pname_raw.endswith("2") else 0

        # Extract state:N parameter (overrides player from name)
        state_m = re.search(r'state:\s*(\d+)', expr)
        if state_m:
            player = int(state_m.group(1)) - 1  # Ludii state:1 = player 0, state:2 = player 1
            expr = expr[:state_m.start()].strip()

        # Evaluate the site set expression
        indices = evaluate_sites(expr, topo)
        if pname in piece_names and indices:
            placements.append((piece_names.index(pname), player, sorted(indices)))

    if not placements:
        # Fallback: auto-place for movement games (even without explicit start placement)
        has_movement = info.has_step or info.has_hop or info.has_slide or "forEach Piece" in info.full_text
        if has_movement and n >= 4 and topo.site_coords:
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
