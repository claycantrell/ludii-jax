#!/usr/bin/env python3
"""
Run comprehensive benchmarks and generate publication-quality plots.

Produces:
1. Batch scaling chart (steps/sec vs batch size, log scale)
2. Game comparison bar chart (throughput by game at optimal batch)
3. Compile time breakdown
"""

import time
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from ludii_jax import compile as ludii_compile


# ── Games to benchmark ──────────────────────────────────────────────

GAMES = {
    "Tic-Tac-Toe\n(3×3)": '(game "T" (players 2) (equipment {(board (square 3)) (piece "M" Each)}) (rules (play (move Add (to (sites Empty)))) (end {(if (is Line 3) (result Mover Win)) (if (is Full) (draw))})))',
    "Gomoku\n(15×15)": '(game "G" (players 2) (equipment {(board (square 15)) (piece "M" Each)}) (rules (play (move Add (to (sites Empty)))) (end (if (is Line 5) (result Mover Win)))))',
    "Reversi\n(8×8)": '(game "R" (players 2) (equipment {(board (square 8)) (piece "D" Each)}) (rules (play (move Add (to (sites Empty)))) (end (if (is Full) (draw)))))',
    "Go\n(9×9)": '(game "Go" (players 2) (equipment {(board (square 9)) (piece "M" Each)}) (rules (play (move Add (to (sites Empty)))) (end (if (no Moves Mover) (result Mover Loss)))))',
    "Hex\n(11×11)": '(game "H" (players 2) (equipment {(board (hex Diamond 11)) (piece "M" Each)}) (rules (play (move Add (to (sites Empty)))) (end (if (is Connected Mover) (result Mover Win)))))',
    "Go\n(19×19)": '(game "Go19" (players 2) (equipment {(board (square 19)) (piece "M" Each)}) (rules (play (move Add (to (sites Empty)))) (end (if (no Moves Mover) (result Mover Loss)))))',
}

BATCH_SIZES = [1, 4, 16, 64, 256, 1024, 4096]

# ── Helpers ──────────────────────────────────────────────────────────

def pick_action(key, legal):
    logits = jnp.where(legal, 0.0, -1e9)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def bench_batched(env, batch_size, num_rounds=50):
    v_init = jax.jit(jax.vmap(env.init))
    v_step = jax.jit(jax.vmap(env.step))

    @jax.jit
    def pick_actions(key, legal_mask):
        logits = jnp.where(legal_mask, 0.0, -1e9)
        keys = jax.random.split(key, legal_mask.shape[0])
        return jax.vmap(lambda k, l: jax.random.categorical(k, l).astype(jnp.int32))(keys, logits)

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    states = v_init(keys)
    dummy = pick_actions(jax.random.PRNGKey(99), states.legal_action_mask)
    _ = v_step(states, dummy)
    jax.block_until_ready(_.legal_action_mask)

    states = v_init(keys)
    key = jax.random.PRNGKey(1)
    t0 = time.perf_counter()
    total = 0
    for _ in range(num_rounds):
        key, sk = jax.random.split(key)
        actions = pick_actions(sk, states.legal_action_mask)
        states = v_step(states, actions)
        total += batch_size
    jax.block_until_ready(states.legal_action_mask)
    return total / (time.perf_counter() - t0)


# ── Style ────────────────────────────────────────────────────────────

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
BG = '#0D1117'
FG = '#C9D1D9'
GRID = '#21262D'
ACCENT = '#58A6FF'


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, which='both')
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(f"JAX: {jax.default_backend()} | {jax.devices()}")
    results = {}

    for name, text in GAMES.items():
        short = name.split('\n')[0]
        print(f"\n  Benchmarking {short}...", end=" ", flush=True)
        try:
            env = ludii_compile(text)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        game_results = {"sites": env.num_sites, "actions": env.num_actions, "batch": {}}
        for bs in BATCH_SIZES:
            rounds = max(5, 200 // bs)
            try:
                sps = bench_batched(env, bs, num_rounds=rounds)
                game_results["batch"][bs] = sps
                print(f"{bs}={sps/1000:.0f}K", end=" ", flush=True)
            except Exception:
                break
        results[name] = game_results
        print()

    # Save raw data
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/data.json", "w") as f:
        json.dump({k: {"sites": v["sites"], "batch": {str(bk): bv for bk, bv in v["batch"].items()}} for k, v in results.items()}, f, indent=2)

    # ── Plot 1: Batch Scaling (log-log) ──────────────────────────────

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    style_ax(ax)

    for i, (name, data) in enumerate(results.items()):
        batches = sorted(data["batch"].keys())
        throughputs = [data["batch"][b] for b in batches]
        ax.plot(batches, throughputs, 'o-', color=COLORS[i % len(COLORS)],
                linewidth=2.5, markersize=8, label=name, zorder=3)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Batch Size (simultaneous games)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (steps/sec)', fontsize=13, fontweight='bold')
    ax.set_title('ludii-jax: Throughput Scaling with Batch Size', fontsize=16, fontweight='bold', pad=15)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

    legend = ax.legend(fontsize=10, loc='upper left', facecolor='#161B22', edgecolor=GRID,
                       labelcolor=FG, framealpha=0.95)

    # Add annotation for peak
    peak_game = max(results.keys(), key=lambda k: max(results[k]["batch"].values()) if results[k]["batch"] else 0)
    peak_val = max(results[peak_game]["batch"].values())
    peak_bs = max(results[peak_game]["batch"].keys(), key=lambda b: results[peak_game]["batch"][b])
    ax.annotate(f'{peak_val/1e6:.1f}M steps/sec',
                xy=(peak_bs, peak_val), xytext=(peak_bs * 0.3, peak_val * 1.5),
                fontsize=11, color=ACCENT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5))

    plt.tight_layout()
    plt.savefig('benchmarks/results/batch_scaling.png', dpi=200, facecolor=BG, bbox_inches='tight')
    print("\n  Saved batch_scaling.png")

    # ── Plot 2: Peak Throughput Bar Chart ────────────────────────────

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    style_ax(ax)

    names = list(results.keys())
    peaks = [max(results[n]["batch"].values()) if results[n]["batch"] else 0 for n in names]
    sites = [results[n]["sites"] for n in names]

    bars = ax.bar(range(len(names)), [p / 1e6 for p in peaks], color=COLORS[:len(names)],
                  edgecolor='white', linewidth=0.5, width=0.7, zorder=3)

    # Add value labels on bars
    for bar, peak, site in zip(bars, peaks, sites):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                f'{peak/1e6:.1f}M', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=FG)
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2,
                f'{site} cells', ha='center', va='center',
                fontsize=9, color='black', fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10, color=FG)
    ax.set_ylabel('Peak Throughput (M steps/sec)', fontsize=13, fontweight='bold')
    ax.set_title('ludii-jax: Peak Throughput by Game (batch 4096, CPU)', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, max(p / 1e6 for p in peaks) * 1.25)

    plt.tight_layout()
    plt.savefig('benchmarks/results/peak_throughput.png', dpi=200, facecolor=BG, bbox_inches='tight')
    print("  Saved peak_throughput.png")

    # ── Plot 3: Speedup from batching ────────────────────────────────

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    style_ax(ax)

    for i, (name, data) in enumerate(results.items()):
        if 1 not in data["batch"]:
            continue
        base = data["batch"][1]
        batches = sorted(data["batch"].keys())
        speedups = [data["batch"][b] / base for b in batches]
        ax.plot(batches, speedups, 'o-', color=COLORS[i % len(COLORS)],
                linewidth=2.5, markersize=8, label=name, zorder=3)

    # Reference line: perfect linear scaling
    ax.plot(BATCH_SIZES, BATCH_SIZES, '--', color=FG, alpha=0.3, linewidth=1, label='Perfect linear')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Batch Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup vs Single Game', fontsize=13, fontweight='bold')
    ax.set_title('ludii-jax: Vectorization Speedup (vmap)', fontsize=16, fontweight='bold', pad=15)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}x'))

    legend = ax.legend(fontsize=10, loc='upper left', facecolor='#161B22', edgecolor=GRID,
                       labelcolor=FG, framealpha=0.95)

    plt.tight_layout()
    plt.savefig('benchmarks/results/speedup.png', dpi=200, facecolor=BG, bbox_inches='tight')
    print("  Saved speedup.png")

    print("\nDone! Charts in benchmarks/results/")


if __name__ == "__main__":
    main()
