# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

ludii-jax compiles [Ludii](https://ludii.games/) `.lud` board game descriptions directly into JAX-accelerated environments. No intermediate format -- the Ludii game description is the source of truth.

## Setup

```bash
pip install -e ".[test]"
```

## Key Commands

**Run trace validation (88/97 pass):**
```bash
python tests/test_against_ludii.py
```

**Run coverage on full Ludii corpus:**
```bash
python tests/test_coverage.py /path/to/ludii/games/expanded 200
```

**Generate reference traces (requires Ludii.jar + Java):**
```bash
python tests/generate_traces.py /path/to/ludii/games/expanded
```

There is no pytest suite yet -- the test scripts are standalone.

## Architecture

The pipeline: `.lud` text -> parse tree -> game info -> JAX functions -> Environment.

```
src/ludii_jax/
  compile.py              # Top-level entry point. Routes to mechanics compilers.
  parser/parse.py         # Lark grammar parser (100% of Ludii)
  parser/ludii_grammar.lark
  analysis/topology.py    # Board spec -> adjacency graph (every board type)
  analysis/game_info.py   # Extract pieces, mechanics, phases from parse tree
  analysis/sites.py       # Recursive site set evaluator (difference, expand, etc.)
  compiler/moves.py       # Step/Hop/Slide/Leap/Place/Sow -> JAX functions
  compiler/effects.py     # Custodial/Surround capture -> JAX functions
  compiler/conditions.py  # Line/Connected/NoMoves end conditions -> JAX
  compiler/compose.py     # Assemble into init/step/legal API
  runtime/state.py        # Dynamic state namedtuple + JAX pytree registration
  runtime/environment.py  # Environment wrapper: init/step with dice, truncation
  runtime/lookup.py       # Precomputed slide/line/edge tables from topology
```

### Key files by size

- `compile.py` (~740 lines) -- main compiler, mechanic detection, multi-phase
- `topology.py` (~590 lines) -- all board constructors
- `moves.py` (~570 lines) -- movement compilers with promotion, chain capture
- `game_info.py` (~310 lines) -- parse tree analysis
- `sites.py` (~240 lines) -- recursive site set evaluation
- `conditions.py` (~180 lines) -- end condition compilers with connection BFS
- `effects.py` (~180 lines) -- capture effect compilers

### JAX Conventions

- All state is JAX pytree-compatible (namedtuple with `_replace`)
- Use dtype constants from `runtime/state.py`: `BOARD_DTYPE = jnp.int8`, `ACTION_DTYPE = jnp.int16`
- Avoid Python loops inside JAX-traced functions; use `jax.lax.fori_loop` or vectorized ops
- The `jax.lax.switch` dispatches between phases/piece types at runtime
- Scatter operations with duplicate indices need `.max()` not `.set()` (the scatter bug)

### Common Patterns

- **Mechanic detection**: check keywords in play section text (not full game text)
- **Per-piece compilation**: loop over `info.pieces`, compile separate functions
- **Direction masks**: `(p1_dirs, p2_dirs)` tuple for per-player forward restriction
- **Phase transition**: `addl_info_fn` callback in environment step
- **Custodial capture**: only check from last-moved-to cell, not globally

### Testing

Reference traces from Ludii's Java engine are in `tests/traces/`. Each is a JSON with move-by-move board states. The validation script replays moves through ludii-jax and compares legal actions and termination.

To add a new test game: add it to `tests/generate_traces.py`'s `TEST_GAMES` dict, run the generator, then run validation.

### Trace files

Trace JSON files are gitignored (generated artifacts). To regenerate:
1. Download `Ludii.jar` from https://ludii.games/
2. Place in `tests/`
3. Run `python tests/generate_traces.py /path/to/ludii/games`
