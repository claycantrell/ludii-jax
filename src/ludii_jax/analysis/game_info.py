"""
Extract game metadata from a Ludii parse tree.

Scans the AST to determine: board spec, piece types, player count,
movement types used, state fields needed, phase structure, end conditions.
"""

import re
from dataclasses import dataclass, field
from lark import Token

from ..parser.parse import find_child, find_all, get_text
from .topology import BoardTopology, build_topology


@dataclass
class PieceInfo:
    name: str
    owner: str  # "P1", "P2", "both"


@dataclass
class GameInfo:
    name: str = "Unknown"
    num_players: int = 2
    topology: BoardTopology = None

    pieces: list = field(default_factory=list)  # [PieceInfo, ...]
    piece_name_map: dict = field(default_factory=dict)  # merged name → canonical
    piece_content: dict = field(default_factory=dict)  # name → raw definition text

    has_set_forward: bool = False
    is_mancala: bool = False
    is_dice: bool = False

    # Detected mechanics (determines state shape)
    has_step: bool = False
    has_hop: bool = False
    has_slide: bool = False
    has_leap: bool = False
    has_sow: bool = False
    has_dice: bool = False
    has_phases: bool = False
    has_hand: bool = False
    has_stacking: bool = False
    max_stack_height: int = 1  # >1 for stacking games
    has_capture: bool = False
    has_flip: bool = False
    has_promote: bool = False
    has_extra_turn: bool = False
    has_score: bool = False
    has_connected: bool = False

    # Primary mechanic: PLACE, MOVEMENT, FOREACH_PIECE, MULTI_PHASE, REMOVE, SELECT, MANCALA, DICE
    mechanic: str = "PLACE"

    # Raw text for pattern matching
    full_text: str = ""
    play_text: str = ""
    start_text: str = ""
    end_text: str = ""

    # Mancala params
    mancala_seeds: int = 4
    dice_count: int = 2
    dice_faces: int = 6


def extract_game_info(tree) -> GameInfo:
    """Extract all game metadata from a Ludii parse tree."""
    info = GameInfo()
    info.full_text = get_text(tree)

    # Name
    info.name = ""
    for c in tree.children:
        if isinstance(c, Token) and c.type == "ESCAPED_STRING":
            info.name = str(c).strip('"')
            break

    # Players
    _extract_players(tree, info)

    # Equipment
    _extract_equipment(tree, info)

    # Detect mancala
    if not info.is_mancala and "sow" in info.full_text.lower() and "track" in info.full_text.lower():
        info.is_mancala = True
        info.has_sow = True

    # Detect dice
    if "dice" in info.full_text.lower() or "Die " in info.full_text:
        info.is_dice = True
        info.has_dice = True
        m = re.search(r'num:(\d+)', info.full_text)
        if m:
            info.dice_count = int(m.group(1))

    # Merge symmetric pieces
    _merge_symmetric_pieces(info)

    # Extract rules sections
    rules = find_child(tree, "rules")
    if rules:
        rules_text = get_text(rules)
        info.start_text = rules_text[:rules_text.find("play")] if "play" in rules_text else ""

        # Detect mechanics from PLAY section + piece definitions (for forEach Piece)
        # NOT from the full text — avoids false positives from unrelated sections
        play_node = None
        rules_content = find_child(rules, "rules_content")
        if rules_content:
            for item in find_all(rules_content, "rules_item"):
                p = find_child(item, "play")
                if p:
                    play_node = p
                    break
        play_mt = get_text(play_node) if play_node else ""
        # For forEach Piece games, movement is defined in piece equipment — include it
        if "forEach Piece" in play_mt:
            equip = find_child(tree, "equipment")
            if equip:
                for item in find_all(equip, "equip_item"):
                    etype = find_child(item, "equip_type")
                    if etype:
                        for c in etype.children:
                            if isinstance(c, Token) and str(c) == "piece":
                                content = find_child(item, "equip_content")
                                if content:
                                    play_mt += " " + get_text(content)
        info.has_step = "move Step" in play_mt or ("Step" in play_mt and "move" in play_mt)
        info.has_hop = "move Hop" in play_mt or ("Hop" in play_mt and "move" in play_mt)
        info.has_slide = "move Slide" in play_mt or ("Slide" in play_mt and "move" in play_mt)
        info.has_leap = "Leap" in play_mt
        # Effects and conditions: detect from rules text (play + end + piece defs)
        info.has_capture = "remove" in play_mt.lower() and ("between" in play_mt or "custodial" in play_mt)
        info.has_flip = "flip" in rules_text.lower()
        info.has_promote = "promote" in rules_text.lower()
        info.has_extra_turn = "moveAgain" in play_mt
        info.has_score = "addScore" in rules_text or "set Score" in rules_text or "by_score" in rules_text.lower()
        info.has_connected = "is Connected" in rules_text
        info.has_phases = "phases" in rules_text.lower() or "phase" in rules_text.lower()
        info.has_hand = "hand" in info.full_text.lower() or "Hand" in info.full_text

        # Stacking detection
        if "stack:True" in info.full_text or "stack:true" in info.full_text:
            info.has_stacking = True
            # Detect max stack height: (< (size Stack at:(site)) N) or just N near Stack
            m_stack = re.search(r'<\s*size Stack.*?(\d+)', info.full_text)
            if not m_stack:
                m_stack = re.search(r'size Stack.*?(\d+)', info.full_text)
            if m_stack:
                info.max_stack_height = int(m_stack.group(1))
            else:
                info.max_stack_height = 8

        # Mancala seed count
        m = re.search(r'set Count (\d+)', rules_text)
        if m:
            info.mancala_seeds = int(m.group(1))

        # Store play text for compile.py (fallback to full rules if no play node)
        info.play_text = play_mt if play_mt else rules_text

    # Classify primary mechanic
    info.mechanic = _classify_mechanic(info)

    return info


def _classify_mechanic(info: GameInfo) -> str:
    """Classify the primary game mechanic from detected features."""
    if info.is_mancala:
        return "MANCALA"
    if info.is_dice:
        return "DICE"

    pt = info.play_text
    has_explicit_phases = "phases:" in pt or "phases:{" in pt.replace(" ", "")
    has_hand = "hand" in info.full_text.lower() and "handSite" in pt

    if has_explicit_phases and has_hand and ("forEach Piece" in pt or "move Step" in pt):
        return "MULTI_PHASE"
    if has_explicit_phases and "move Remove" in pt and ("Hop" in pt or "move Step" in pt or "forEach Piece" in pt):
        return "MULTI_PHASE_REMOVE"
    if "move Add" in pt or "move Claim" in pt:
        return "PLACE"
    if "satisfy" in pt:
        return "PLACE"
    if "move Remove" in pt and "forEach Piece" not in pt:
        return "REMOVE"
    if "move Select" in pt and "forEach Piece" not in pt and "move Step" not in pt:
        return "SELECT"
    if "forEach Site" in pt and "forEach Piece" not in pt:
        return "PLACE"
    if "handSite" in pt and "forEach Piece" not in pt:
        return "PLACE"
    if "forEach Piece" in pt:
        return "FOREACH_PIECE"
    if "move Step" in pt or "move Hop" in pt or "move Slide" in pt:
        return "MOVEMENT"
    return "PLACE"


def _extract_players(tree, info: GameInfo):
    players = find_child(tree, "players")
    if players:
        content = get_text(players)
        if "player N" in content or "player S" in content:
            info.has_set_forward = True
        # Count players
        player_count = len(re.findall(r'player\s+[NSEW]', content))
        if player_count > 2:
            info.num_players = player_count
        nums = re.findall(r'players\s+(\d+)', content)
        if nums:
            info.num_players = int(nums[0])


def _extract_equipment(tree, info: GameInfo):
    equip = find_child(tree, "equipment")
    if not equip:
        return

    board_text = ""
    for item in find_all(equip, "equip_item"):
        etype = find_child(item, "equip_type")
        if not etype:
            continue
        type_str = ""
        for c in etype.children:
            if isinstance(c, Token):
                type_str = str(c)
                break

        content = find_child(item, "equip_content")
        content_str = get_text(content) if content else ""

        if type_str == "board":
            board_text = content_str
        elif type_str in ("mancalaBoard", "surakartaBoard"):
            info.is_mancala = True
            info.has_sow = True
            # Parse mancala board dimensions
            nums = [int(t) for t in content_str.split() if t.isdigit()]
            has_stores = "store:None" not in content_str and "store:none" not in content_str
            store_count = 2 if has_stores else 0
            if len(nums) >= 2:
                rows, cols = nums[0], nums[1]
                total_cells = rows * cols + store_count
                # Model as rectangle with extra columns for stores
                board_text = f"rectangle {cols + (store_count // rows if rows > 0 else 0)} {rows}"
            elif nums:
                board_text = f"rectangle {nums[0] + store_count} 2"
            else:
                board_text = f"rectangle {6 + store_count} 2"
        elif type_str == "hand":
            info.has_hand = True
        elif type_str == "piece":
            _parse_piece(content_str, info)
        elif type_str == "die" or type_str == "dice":
            info.is_dice = True
            info.has_dice = True

    if board_text:
        info.topology = build_topology(board_text)
    else:
        info.topology = build_topology("square 8")

    # Extract named regions (regions P1/P2/...) and add to topology
    from .sites import evaluate_sites
    for item in find_all(equip, "equip_item"):
        etype = find_child(item, "equip_type")
        if not etype:
            continue
        type_str = ""
        for c in etype.children:
            if isinstance(c, Token):
                type_str = str(c)
                break
        if type_str == "regions":
            content = find_child(item, "equip_content")
            content_str = get_text(content) if content else ""
            # Parse: "P1 expand sites Bottom" or "P2 sites Top"
            parts = content_str.split(None, 1)
            if len(parts) >= 2:
                region_name = parts[0].lower()  # "p1", "p2"
                region_expr = parts[1]
                try:
                    cells = evaluate_sites(region_expr, info.topology)
                    if cells:
                        import numpy as np
                        mask = np.zeros(info.topology.num_sites, dtype=bool)
                        for c in cells:
                            mask[c] = True
                        info.topology.regions[region_name] = mask
                except Exception:
                    pass


def _parse_piece(content: str, info: GameInfo):
    tokens = content.strip().split()
    if not tokens:
        return

    name = re.sub(r'\d+$', '', tokens[0].strip('"').lower())
    if not name:
        name = tokens[0].strip('"').lower()

    owner = "both"
    for t in tokens[1:]:
        if t == "Each":
            owner = "both"; break
        elif t == "P1":
            owner = "P1"; break
        elif t == "P2":
            owner = "P2"; break
        elif t in ("Neutral", "Shared"):
            owner = "both"; break

    # Store piece content for merge keyword checking
    info.piece_content[name] = content

    # Deduplicate
    for i, p in enumerate(info.pieces):
        if p.name == name:
            if p.owner != owner:
                info.pieces[i] = PieceInfo(name, "both")
            return

    info.pieces.append(PieceInfo(name, owner))


def _merge_symmetric_pieces(info: GameInfo):
    if len(info.pieces) < 2:
        return

    p1 = [(i, p.name) for i, p in enumerate(info.pieces) if p.owner == "P1"]
    p2 = [(i, p.name) for i, p in enumerate(info.pieces) if p.owner == "P2"]
    if not p1 or not p2:
        return

    def _movement_keywords(name):
        content = info.piece_content.get(name, "")
        return {kw for kw in ["Step", "Hop", "Slide", "Leap", "Forward", "Orthogonal"]
                if kw in content}

    merged = set()
    for p1_idx, p1_name in p1:
        for p2_idx, p2_name in p2:
            if p2_idx in merged:
                continue
            # Only merge if movement keywords match
            if _movement_keywords(p1_name) == _movement_keywords(p2_name):
                info.pieces[p1_idx] = PieceInfo(p1_name, "both")
                info.piece_name_map[p2_name] = p1_name
                merged.add(p2_idx)
                break

    for idx in sorted(merged, reverse=True):
        info.pieces.pop(idx)
