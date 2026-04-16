#!/usr/bin/env python3
"""
Benchmark ludii-jax throughput: steps/second for various games and batch sizes.

Measures:
1. Single-game sequential steps/sec (comparable to Ludii single-thread)
2. Batched vmap steps/sec (the JAX advantage)
3. Compile time (JIT overhead, paid once)

Usage: python benchmarks/throughput.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
from ludii_jax import compile


GAMES = {
    "Tic-Tac-Toe": """(game "Tic-Tac-Toe" (players 2) (equipment {(board (square 3)) (piece "Marker" Each)}) (rules (play (move Add (to (sites Empty)))) (end {(if (is Line 3) (result Mover Win)) (if (is Full) (draw))})))""",

    "Hex 11": """(game "Hex" (players 2) (equipment {(board (hex Diamond 11)) (piece "Marker" Each)}) (rules (play (move Add (to (sites Empty)))) (end (if (is Connected Mover) (result Mover Win)))))""",

    "Go 9x9": """(game "Go" (players 2) (equipment {(board (square 9)) (piece "Marker" Each)}) (rules (play (move Add (to (sites Empty)))) (end (if (no Moves Mover) (result Mover Loss)))))""",

    "Reversi": """(game "Reversi" (players 2) (equipment {(board (square 8)) (piece "Disc" Each)}) (rules (play (move Add (to (sites Empty)))) (end (if (is Full) (draw)))))""",
}

BATCH_SIZES = [1, 32, 128, 512, 1024]


def random_legal_action(key, legal_mask):
    """Pick a random legal action."""
    logits = jnp.where(legal_mask, 0.0, -1e9)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def benchmark_single(env, num_steps=500):
    """Benchmark single-game sequential play (JIT-compiled step)."""
    step_fn = jax.jit(env.step)
    init_fn = jax.jit(env.init)

    key = jax.random.PRNGKey(42)
    state = init_fn(key)

    # Warm up JIT
    key, subkey = jax.random.split(key)
    action = random_legal_action(subkey, state.legal_action_mask)
    state_warm = step_fn(state, action)
    jax.block_until_ready(state_warm.legal_action_mask)

    # Benchmark
    state = init_fn(key)
    t0 = time.perf_counter()
    steps = 0
    for _ in range(num_steps):
        if bool(state.terminated):
            key, subkey = jax.random.split(key)
            state = init_fn(subkey)
        key, subkey = jax.random.split(key)
        action = random_legal_action(subkey, state.legal_action_mask)
        state = step_fn(state, action)
        steps += 1
    jax.block_until_ready(state.legal_action_mask)
    elapsed = time.perf_counter() - t0
    return steps / elapsed


def benchmark_batched(env, batch_size, num_rounds=100):
    """Benchmark batched vmap play."""
    v_init = jax.jit(jax.vmap(env.init))
    v_step = jax.jit(jax.vmap(env.step))

    @jax.jit
    def pick_actions(key, legal_mask):
        logits = jnp.where(legal_mask, 0.0, -1e9)
        keys = jax.random.split(key, legal_mask.shape[0])
        return jax.vmap(lambda k, l: jax.random.categorical(k, l).astype(jnp.int32))(keys, logits)

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    states = v_init(keys)

    # Warm up
    key = jax.random.PRNGKey(99)
    dummy = pick_actions(key, states.legal_action_mask)
    _ = v_step(states, dummy)
    jax.block_until_ready(_.legal_action_mask)

    # Benchmark
    states = v_init(keys)
    key = jax.random.PRNGKey(1)
    t0 = time.perf_counter()
    total_steps = 0
    for _ in range(num_rounds):
        key, subkey = jax.random.split(key)
        actions = pick_actions(subkey, states.legal_action_mask)
        states = v_step(states, actions)
        total_steps += batch_size
    jax.block_until_ready(states.legal_action_mask)
    elapsed = time.perf_counter() - t0
    return total_steps / elapsed


def benchmark_compile_time(game_text):
    """Measure compile + init + first step time."""
    t0 = time.perf_counter()
    env = compile(game_text)
    t_compile = time.perf_counter() - t0

    t1 = time.perf_counter()
    state = env.init(jax.random.PRNGKey(0))
    jax.block_until_ready(state.legal_action_mask)
    t_init = time.perf_counter() - t1

    t2 = time.perf_counter()
    action = random_legal_action(jax.random.PRNGKey(1), state.legal_action_mask)
    state = env.step(state, int(action))
    jax.block_until_ready(state.legal_action_mask)
    t_first_step = time.perf_counter() - t2

    return t_compile, t_init, t_first_step, env


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    for game_name, game_text in GAMES.items():
        print(f"{'='*60}")
        print(f"  {game_name}")
        print(f"{'='*60}")

        # Compile time
        t_compile, t_init, t_first, env = benchmark_compile_time(game_text)
        print(f"  Board: {env.num_sites} sites, {env.num_actions} actions")
        print(f"  Compile: {t_compile:.3f}s  Init: {t_init:.3f}s  First step (JIT): {t_first:.3f}s")
        print()

        # Single game
        single_sps = benchmark_single(env, num_steps=1000)
        print(f"  Single game:  {single_sps:>10,.0f} steps/sec")

        # Batched
        for bs in BATCH_SIZES:
            rounds = max(10, 500 // bs)
            batched_sps = benchmark_batched(env, bs, num_rounds=rounds)
            speedup = batched_sps / single_sps
            print(f"  Batch {bs:>5d}:  {batched_sps:>10,.0f} steps/sec  ({speedup:.1f}x)")

        print()


if __name__ == "__main__":
    main()
