#!/usr/bin/env python3
"""Play a random game of any Ludii game using ludii-jax."""

import jax
import jax.numpy as jnp
from ludii_jax import compile


GAME = """
(game "Hex"
    (players 2)
    (equipment {
        (board (hex Diamond 11))
        (piece "Marker" Each)
    })
    (rules
        (play (move Add (to (sites Empty))))
        (end (if (is Connected Mover) (result Mover Win)))
    ))
"""


def play_random(game_text: str, seed: int = 42):
    env = compile(game_text)
    key = jax.random.PRNGKey(seed)

    state = env.init(key)
    step = 0

    print(f"Game compiled: {env.num_sites} sites, {env.num_actions} actions")
    print(f"Legal moves at start: {int(state.legal_action_mask.sum())}")
    print()

    while not state.terminated and step < 500:
        key, subkey = jax.random.split(key)
        legal = jnp.where(state.legal_action_mask, 1.0, -1e9)
        action = jax.random.categorical(subkey, legal).astype(jnp.int32)

        player = int(state.current_player) + 1
        if env.num_actions == env.num_sites:
            print(f"Move {step}: P{player} places at {int(action)}")
        else:
            src, dst = int(action) // env.num_sites, int(action) % env.num_sites
            print(f"Move {step}: P{player} moves {src} -> {dst}")

        state = env.step(state, int(action))
        step += 1

    if state.terminated:
        winners = state.winners
        if (winners > 0).any():
            winner = int(jnp.argmax(winners)) + 1
            print(f"\nP{winner} wins after {step} moves!")
        else:
            print(f"\nDraw after {step} moves.")
    else:
        print(f"\nGame truncated at {step} moves.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            play_random(f.read())
    else:
        play_random(GAME)
