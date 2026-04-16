#!/usr/bin/env python3
"""Batch simulate 1024 games in parallel using vmap."""

import time
import jax
import jax.numpy as jnp
from ludii_jax import compile


GAME = """
(game "Tic-Tac-Toe"
    (players 2)
    (equipment {(board (square 3)) (piece "Marker" Each)})
    (rules
        (play (move Add (to (sites Empty))))
        (end {
            (if (is Line 3) (result Mover Win))
            (if (is Full) (draw))
        })
    ))
"""


def batch_simulate(game_text: str, batch_size: int = 1024, max_steps: int = 200):
    env = compile(game_text)
    print(f"Compiled: {env.num_sites} sites, {env.num_actions} actions")

    # Vectorize init and step
    v_init = jax.jit(jax.vmap(env.init))
    v_step = jax.jit(jax.vmap(env.step))

    # Initialize batch
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    states = v_init(keys)

    # Warm up JIT
    key = jax.random.PRNGKey(99)
    dummy_actions = jax.random.randint(key, (batch_size,), 0, env.num_actions)
    _ = v_step(states, dummy_actions)

    # Simulate
    t0 = time.time()
    total_steps = 0
    key = jax.random.PRNGKey(1)

    for step in range(max_steps):
        key, subkey = jax.random.split(key)
        # Random legal actions
        legal = jnp.where(states.legal_action_mask, 1.0, -1e9)
        actions = jax.random.categorical(
            jax.random.split(subkey, batch_size), legal
        ).astype(jnp.int32)

        states = v_step(states, actions)
        total_steps += batch_size

        terminated = states.terminated.sum()
        if terminated == batch_size:
            break

    elapsed = time.time() - t0
    steps_per_sec = total_steps / elapsed

    print(f"\n{batch_size} games, {step + 1} rounds")
    print(f"Total steps: {total_steps:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {steps_per_sec:,.0f} steps/sec")

    # Results
    wins = (states.winners > 0).any(axis=-1)
    p1_wins = (states.winners[:, 0] > 0).sum()
    p2_wins = (states.winners[:, 1] > 0).sum()
    draws = batch_size - int(p1_wins) - int(p2_wins)
    print(f"\nP1 wins: {int(p1_wins)}, P2 wins: {int(p2_wins)}, Draws: {draws}")


if __name__ == "__main__":
    batch_simulate(GAME)
