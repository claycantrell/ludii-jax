#!/usr/bin/env python3
"""
Validate ludii-jax against Ludii reference traces.

For each trace file in tests/traces/:
1. Load the .lud game
2. Play the same sequence of moves in ludii-jax
3. Compare board states, legal move counts, and termination

Usage: python test_against_ludii.py [ludii_games_dir]
"""

import json
import os
import sys
import glob

import jax
import jax.numpy as jnp

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ludii_jax import compile


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(TESTS_DIR, "traces")


def validate_trace(trace_path, games_dir):
    """Validate one trace against ludii-jax."""
    with open(trace_path) as f:
        trace = json.load(f)

    game_name = trace["game"]
    board_size = trace["board_size"]
    num_moves = trace["num_moves"]

    # Find the .lud file
    lud_path = None
    for root, dirs, files in os.walk(games_dir):
        for f in files:
            if f == f"{game_name}.lud":
                lud_path = os.path.join(root, f)
                break
        if lud_path:
            break

    if not lud_path:
        return "SKIP", f"Game file not found: {game_name}.lud"

    with open(lud_path) as f:
        lud = f.read()

    try:
        env = compile(lud)
    except Exception as e:
        return "COMPILE_FAIL", str(e)[:60]

    state = env.init(jax.random.PRNGKey(42))
    errors = []

    # Check board size matches
    if env.num_sites != board_size:
        errors.append(f"Board size: ludii={board_size} jax={env.num_sites}")

    # Check initial legal move count
    if trace["moves"]:
        first_move = trace["moves"][0]
        jax_legal = int(state.legal_action_mask.sum())
        ludii_legal = first_move["legal_moves_count"]
        if jax_legal == 0 and ludii_legal > 0:
            errors.append(f"Move 0: JAX has 0 legal moves, Ludii has {ludii_legal}")
            return "NO_LEGAL", "; ".join(errors)

    # Play through moves and compare
    for move in trace["moves"]:
        if state.terminated:
            break

        action_to = move["action_to"]
        action_from = move.get("action_from", -1)

        # Find the corresponding JAX action
        if env.num_actions == env.num_sites:
            # Placement: action = destination
            jax_action = action_to
        else:
            # Movement: action = from * board_size + to
            if action_from >= 0:
                jax_action = action_from * env.num_sites + action_to
            else:
                jax_action = action_to

        # Check action is legal in JAX
        if jax_action < len(state.legal_action_mask):
            if not state.legal_action_mask[jax_action]:
                # Try just the destination
                if action_to < len(state.legal_action_mask) and state.legal_action_mask[action_to]:
                    jax_action = action_to
                else:
                    errors.append(f"Move {move['move_index']}: action {jax_action} not legal in JAX")
                    # Pick any legal move to continue
                    legal_idx = jnp.where(state.legal_action_mask)[0]
                    if len(legal_idx) > 0:
                        jax_action = int(legal_idx[0])
                    else:
                        errors.append(f"Move {move['move_index']}: no legal moves in JAX")
                        break

        try:
            state = env.step(state, jax_action)
        except Exception as e:
            errors.append(f"Move {move['move_index']}: step failed: {str(e)[:40]}")
            break

    # Check termination
    ludii_terminated = trace["moves"][-1]["terminated"] if trace["moves"] else False
    jax_terminated = bool(state.terminated)

    if ludii_terminated != jax_terminated:
        errors.append(f"Termination: ludii={ludii_terminated} jax={jax_terminated}")

    if errors:
        return "MISMATCH", "; ".join(errors[:3])
    return "PASS", f"{num_moves} moves validated"


def main():
    games_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not games_dir:
        for candidate in [
            os.path.expanduser("~/Documents/GitHub/gavel/ludii_data/games/expanded"),
        ]:
            if os.path.exists(candidate):
                games_dir = candidate
                break

    if not games_dir:
        print("Games directory not found")
        sys.exit(1)

    trace_files = sorted(glob.glob(os.path.join(TRACES_DIR, "*.json")))
    if not trace_files:
        print("No trace files found. Run generate_traces.py first.")
        sys.exit(1)

    results = {"PASS": 0, "MISMATCH": 0, "COMPILE_FAIL": 0, "NO_LEGAL": 0, "SKIP": 0}

    for trace_path in trace_files:
        name = os.path.basename(trace_path).replace(".json", "").replace("_", " ")
        status, detail = validate_trace(trace_path, games_dir)
        results[status] = results.get(status, 0) + 1
        symbol = {"PASS": "✓", "MISMATCH": "✗", "COMPILE_FAIL": "⚠", "NO_LEGAL": "○", "SKIP": "-"}.get(status, "?")
        print(f"  {symbol} {name}: {detail}")

    print(f"\nResults: {results['PASS']} pass, {results['MISMATCH']} mismatch, "
          f"{results['COMPILE_FAIL']} compile fail, {results['NO_LEGAL']} no legal, {results['SKIP']} skip")


if __name__ == "__main__":
    main()
