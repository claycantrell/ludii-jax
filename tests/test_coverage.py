#!/usr/bin/env python3
"""
Coverage test: compile every Ludii game and check it has legal moves.

Usage: python test_coverage.py <ludii_games_dir> [sample_size]
"""

import os
import signal
import sys

import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ludii_jax import compile


def main():
    games_dir = sys.argv[1] if len(sys.argv) > 1 else None
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not games_dir:
        for candidate in [
            os.path.expanduser("~/Documents/GitHub/gavel/ludii_data/games/expanded"),
        ]:
            if os.path.exists(candidate):
                games_dir = candidate
                break

    if not games_dir:
        print("Usage: python test_coverage.py <ludii_games_dir> [sample_size]")
        sys.exit(1)

    game_files = []
    for root, dirs, files in os.walk(games_dir):
        for f in files:
            if f.endswith('.lud'):
                game_files.append(os.path.join(root, f))

    if sample_size:
        import random
        random.seed(42)
        random.shuffle(game_files)
        game_files = game_files[:sample_size]
    else:
        game_files.sort()

    ok = 0
    total = 0
    fails = []

    for path in game_files:
        total += 1
        name = os.path.basename(path)
        with open(path) as fh:
            lud = fh.read().strip()
        try:
            signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(30)
            env = compile(lud)
            state = env.init(jax.random.PRNGKey(0))
            signal.alarm(0)
            if state.legal_action_mask.sum() > 0:
                ok += 1
            else:
                fails.append((name, "no_legal"))
        except TimeoutError:
            signal.alarm(0)
            fails.append((name, "timeout"))
        except Exception as e:
            signal.alarm(0)
            fails.append((name, str(e).split('\n')[0][:50]))

        if total % 200 == 0:
            print(f"  {ok}/{total} ({ok/total*100:.0f}%)...", flush=True)

    pct = ok / total * 100 if total > 0 else 0
    print(f"\n{ok}/{total} ({pct:.1f}%)")

    if fails:
        print(f"\nFailing ({len(fails)}):")
        for name, err in fails[:30]:
            print(f"  {name}: {err}")
        if len(fails) > 30:
            print(f"  ... +{len(fails) - 30} more")

    # Exit with error if below threshold
    if pct < 95:
        sys.exit(1)


if __name__ == "__main__":
    main()
