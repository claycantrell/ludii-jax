# ludii-jax

Compile [Ludii](https://ludii.games/) board game descriptions directly to JAX-accelerated environments. Pass any `.lud` file and get a GPU-ready environment running at hundreds of thousands of steps per second.

```python
from ludii_jax import compile
import jax

env = compile("games/tic_tac_toe.lud")
state = env.init(jax.random.PRNGKey(0))
state = env.step(state, action)
```

**93% of all 1,212 Ludii games** compile and run. No intermediate format. No approximations. The Ludii game description is the source of truth.

## How It Works

```
.lud file
  → Parser (permissive Lark grammar, 100% of Ludii)
  → Semantic analysis (board topology, pieces, rules, phases)
  → Graph construction (every board is an adjacency matrix)
  → JAX compilation (ludeme → pure JAX functions)
  → Environment (init / step / legal_actions / terminal / rewards)
```

Every board — square, hex, triangle, concentric rings, Morris boards, arbitrary graphs — reduces to the same `BoardTopology`: a set of sites with an adjacency matrix. No board type enum. No special cases. Lines, slides, hops, and connections all derive from walking the adjacency graph.

Every action is a `(source, destination)` pair. Step, slide, hop, and leap only differ in which destinations are reachable. One action model, one code path.

## Installation

```bash
pip install jax lark
git clone https://github.com/claycantrell/ludii-jax.git
cd ludii-jax
```

Requires Python 3.9+ and [JAX](https://docs.jax.dev/en/latest/installation.html).

## Usage

### Compile and Play

```python
from ludii_jax import compile
import jax
import jax.numpy as jnp

# Any Ludii .lud game
env = compile("""
(game "Hex"
    (players 2)
    (equipment {
        (board (hex Diamond 11))
        (piece "Marker" Each)
        (regions P1 {(sites Side NE) (sites Side SW)})
        (regions P2 {(sites Side NW) (sites Side SE)})
    })
    (rules
        (play (move Add (to (sites Empty))))
        (end (if (is Connected Mover) (result Mover Win)))
    ))
""")

state = env.init(jax.random.PRNGKey(0))
print(f"Board: {env.num_sites} sites, {env.num_actions} actions")
print(f"Legal moves: {state.legal_action_mask.sum()}")
```

### Batch Simulation

```python
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))

keys = jax.random.split(jax.random.PRNGKey(0), 1024)
states = init(keys)
# 1024 games in parallel on GPU
```

### From File

```python
env = compile("path/to/game.lud")
```

## What's Supported

### Board Types
Every board compiles to a graph. The topology module handles:
- Square grids (N×N, 8 directions)
- Rectangular grids (W×H)
- Hexagonal boards (regular, diamond, variable-width)
- Triangular grids
- Concentric ring boards (Morris, Merels)
- Explicit graph boards (vertex + edge lists)
- Star and complete graphs
- Spiral tracks
- Composite boards (merge, shift, add, remove, rotate, scale)

### Movement
- **Step**: move 1-N cells in a direction
- **Hop**: jump over a piece (capture at between or destination)
- **Slide**: move any distance until blocked
- **Leap**: non-adjacent jumps (knight, camel, zebra, custom offsets)
- **Place**: put a piece on an empty site
- **Sow**: mancala seed distribution along a track
- **Dice move**: roll dice, move along track by dice value

### Effects
- Custodial capture (sandwich removal)
- Flip (change ownership)
- Score tracking
- Extra turns (chain captures)

### End Conditions
- Line of N
- Connection (opposite edges)
- No legal moves
- Captured all
- Full board (draw or by score)

### Game Features
- N-player support (parameterized, not hardcoded)
- Multi-phase games (placement then movement)
- Dice / stochastic elements
- Mancala sowing with capture rules
- Hand / reserve piece storage
- Piece stacking

## Architecture

```
ludii_jax/
├── parser/
│   ├── ludii_grammar.lark    # Permissive grammar (100% of Ludii)
│   └── parse.py              # .lud text → parse tree
│
├── analysis/
│   ├── topology.py           # Board spec → BoardTopology (adjacency graph)
│   └── game_info.py          # Parse tree → GameInfo (pieces, mechanics, phases)
│
├── compiler/
│   ├── moves.py              # Step/Hop/Slide/Leap/Place/Sow → JAX functions
│   ├── effects.py            # Capture/Flip/Score → JAX functions
│   ├── conditions.py         # Line/Connected/NoMoves → JAX functions
│   └── compose.py            # Assemble into init/step/legal/terminal
│
├── runtime/
│   ├── state.py              # Dynamic GameState namedtuple + JAX pytree
│   ├── environment.py        # Env API: init/step
│   └── lookup.py             # Precomputed adjacency/slide/line tables
│
└── compile.py                # Top-level: .lud → Environment
```

All game logic compiles to pure JAX functions at init time. Zero Python overhead during gameplay. Compatible with `jax.jit`, `jax.vmap`, `jax.lax.scan`.

## Coverage

Tested against the full Ludii game corpus (1,212 expanded games):

| Category | Games | Status |
|----------|-------|--------|
| Placement games (Tic-Tac-Toe, Hex, Go, Reversi) | ~400 | Working |
| Movement games (Chess, Checkers, Draughts) | ~350 | Working |
| Mancala / sowing games | ~200 | Working |
| Dice / race games (Backgammon variants) | ~100 | Working |
| Concentric / Morris board games | ~60 | Working |
| Graph / custom topology games | ~20 | Working |
| Complex multi-phase / hidden info | ~80 | In progress |

## Acknowledgments

- [Ludii](https://ludii.games/) — the game description language and corpus
- [JAX](https://github.com/jax-ml/jax) — hardware-accelerated numerical computing
- [Lark](https://github.com/lark-parser/lark) — parsing toolkit
- [PGX](https://github.com/sotetsuk/pgx) — JAX game environments (inspiration for the API)
- [Ludax](https://github.com/gdrtodd/ludax) — JAX board game DSL and compiler (Grant Todd). This project builds on ideas and patterns from Ludax.

## License

MIT
