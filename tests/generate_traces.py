#!/usr/bin/env python3
"""
Generate reference traces from Ludii for a set of test games.

Usage: python generate_traces.py [ludii_games_dir]

Compiles the Java trace generator, runs it against each test game,
and saves JSON traces to tests/traces/.
"""

import json
import os
import subprocess
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LUDII_JAR = os.path.join(TESTS_DIR, "Ludii.jar")
TRACES_DIR = os.path.join(TESTS_DIR, "traces")
JAVA_SRC = os.path.join(TESTS_DIR, "generate_trace.java")

# Key games to validate — cover different mechanics
TEST_GAMES = {
    # Placement games
    "Tic-Tac-Toe": "board/space/line/Tic-Tac-Toe.lud",
    "Gomoku": "board/space/line/Gomoku.lud",
    "Hex": "board/space/connection/Hex.lud",
    "Yavalath": "board/space/line/Yavalath.lud",
    "Reversi": "board/space/territory/Reversi.lud",

    # Movement games
    "English Draughts": "board/war/leaping/diagonal/English Draughts.lud",
    "Breakthrough": "board/war/replacement/eliminate/all/Breakthrough.lud",
    "Wolf and Sheep": "board/hunt/Wolf and Sheep.lud",

    # Mancala
    "Oware": "board/sow/two_rows/Oware.lud",
    "Kalah": "board/sow/two_rows/Kalah.lud",

    # Connection
    "Y": "board/space/connection/Y (Hex).lud",

    # Line games
    "Connect-Four": "board/space/line/Connect-Four.lud",
    "Pente": "board/space/line/Pente.lud",
}


def compile_java():
    """Compile the trace generator."""
    print("Compiling trace generator...")
    result = subprocess.run(
        ["javac", "-cp", LUDII_JAR, JAVA_SRC],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Compile failed: {result.stderr}")
        return False
    return True


def generate_trace(game_path, seed=42, max_moves=100):
    """Run the trace generator for one game."""
    result = subprocess.run(
        ["java", "-cp", f"{TESTS_DIR}:{LUDII_JAR}", "generate_trace",
         game_path, str(seed), str(max_moves)],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def main():
    games_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not games_dir:
        # Try to find Ludii games
        for candidate in [
            os.path.expanduser("~/Documents/GitHub/gavel/ludii_data/games/expanded"),
            "/tmp/ludii_games",
        ]:
            if os.path.exists(candidate):
                games_dir = candidate
                break

    if not games_dir or not os.path.exists(games_dir):
        print(f"Games directory not found. Usage: python generate_traces.py <ludii_games_dir>")
        sys.exit(1)

    if not compile_java():
        sys.exit(1)

    os.makedirs(TRACES_DIR, exist_ok=True)

    for name, rel_path in TEST_GAMES.items():
        game_path = os.path.join(games_dir, rel_path)
        if not os.path.exists(game_path):
            print(f"  SKIP {name}: {rel_path} not found")
            continue

        print(f"  Generating trace for {name}...", end=" ")
        trace = generate_trace(game_path, seed=42, max_moves=50)

        if trace:
            safe_name = name.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(TRACES_DIR, f"{safe_name}.json")
            with open(out_path, "w") as f:
                json.dump(trace, f, indent=2)
            print(f"OK ({trace['num_moves']} moves)")
        else:
            print("FAILED")


if __name__ == "__main__":
    main()
