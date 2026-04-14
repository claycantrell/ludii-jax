# ludii-jax

Compile [Ludii](https://ludii.games/) board game descriptions directly to JAX-accelerated environments. Pass any `.lud` file and get a GPU-ready environment running at hundreds of thousands of steps per second.

```python
from ludii_jax import compile
import jax

env = compile("games/tic_tac_toe.lud")
state = env.init(jax.random.PRNGKey(0))
state = env.step(state, action)
```

**96% of all 1,212 Ludii games** compile and produce legal moves. **67 games** validated move-for-move against Ludii reference traces with zero divergence.

## How It Works

```
.lud file
  -> Parser (permissive Lark grammar, 100% of Ludii)
  -> Semantic analysis (board topology, pieces, rules, phases)
  -> Graph construction (every board is an adjacency matrix)
  -> JAX compilation (ludeme -> pure JAX functions)
  -> Environment (init / step / legal_actions / terminal / rewards)
```

Every board -- square, hex, triangle, concentric rings, Morris boards, arbitrary graphs -- reduces to the same `BoardTopology`: a set of sites with an adjacency matrix. Lines, slides, hops, and connections all derive from walking the adjacency graph.

Every action is a `(source, destination)` pair. Step, slide, hop, and leap only differ in which destinations are reachable. One action model, one code path.

## Installation

```bash
pip install jax jaxlib lark numpy
git clone https://github.com/claycantrell/ludii-jax.git
cd ludii-jax
pip install -e .
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

## Validated Games

67 games validated move-for-move against Ludii reference traces (50 moves each, zero divergence):

**Placement / Line:** Tic-Tac-Toe, Gomoku, Pente, Connect6, Yavalath, Squava, Notakto, Tic-Tac-Toe Misere, Tapatan, Picaria, Nine Holes, Three Men's Morris, Alquerque de Tres, Driesticken, Dris at-Talata, San-Noku-Narabe, Selbia, Ngre E E, Ngrin, Engijn Zirge, Djara-Badakh, Akidada, Roll-Ing to Four, Fanorona Telo, Wure Dune, Yavalax

**Connection:** Hex, Y, Havannah, Cross, Crossway, Gonnect, Unlur, Master Y, Chameleon, Diagonal Hex, Esa Hex, Gyre, Pippinzip, Scaffold, ConHex

**Territory:** Reversi, Go, Weiqi, Atari Go, BlooGo, One-Eyed Go, Rolit, Cavity, Flower Shop, Redstone, Patok, Mity, Dorvolz, HexTrike, MacBeth

**Movement:** English Draughts (with promotion + chain capture), Wolf and Sheep, Breakthrough, Clobber, Jeson Mor, Bamboo

**Custodial:** Hasami Shogi, Dai Hasami Shogi

**Mancala:** Oware, Kalah

**Other:** Ecosys

## What's Supported

### Board Types
- Square grids (N x N, 8 directions)
- Rectangular grids (W x H)
- Hexagonal boards (regular, diamond, triangle, variable-width, rectangle)
- Triangular grids
- Concentric ring boards (Nine Men's Morris) with Ludii-matching vertex numbering
- Explicit graph boards (vertex + edge lists)
- Star, complete, spiral graphs
- Composite boards (merge, shift, add, remove, rotate, scale)

### Movement
- **Step**: move 1-N cells in a direction (per-player forward restriction)
- **Hop**: jump over a piece (opponent, friendly, or any occupied)
- **Slide**: move any distance until blocked (direction + cell blocking)
- **Leap**: non-adjacent jumps (knight, camel, custom offsets)
- **Place**: put a piece on an empty site
- **Sow**: mancala seed distribution with per-player tracks and store skipping

### Effects
- Custodial capture (from last-moved-to, orthogonal/all directions)
- Surround capture (corner cells)
- Piece promotion (Counter -> DoubleCounter at opposite row)
- Chain capture (moveAgain with forced continuation)
- Forced capture priority (hops take precedence over steps)
- Score tracking

### End Conditions
- Line of N (with region exclusion for starting rows)
- Connection (flood-fill BFS between board sides)
- No legal moves
- Piece count threshold
- Full board (draw or by score)

### Game Features
- N-player support (parameterized, not hardcoded)
- Multi-phase games (placement -> movement with phase transition)
- Per-player direction masks (forward diagonal for P1, backward diagonal for P2)
- Per-piece direction and blocking (king vs regular pieces)
- Dice / stochastic elements
- Mancala sowing with Kalah capture rules
- Recursive site set evaluation (difference, union, intersection, expand)
- Named player regions extracted from equipment

## Architecture

```
ludii_jax/
  parser/
    ludii_grammar.lark    # Permissive grammar (100% parse rate)
    parse.py              # .lud text -> parse tree

  analysis/
    topology.py           # Board spec -> BoardTopology (adjacency graph)
    game_info.py          # Parse tree -> GameInfo (pieces, mechanics, phases)
    sites.py              # Recursive site set evaluator

  compiler/
    moves.py              # Step/Hop/Slide/Leap/Place/Sow -> JAX functions
    effects.py            # Custodial/Surround/Score -> JAX functions
    conditions.py         # Line/Connected/NoMoves -> JAX functions
    compose.py            # Assemble into init/step/legal/terminal

  runtime/
    state.py              # Dynamic GameState namedtuple + JAX pytree
    environment.py        # Env API: init/step
    lookup.py             # Precomputed adjacency/slide/line tables

  compile.py              # Top-level: .lud -> Environment
```

All game logic compiles to pure JAX functions at init time. Zero Python overhead during gameplay. Compatible with `jax.jit`, `jax.vmap`, `jax.lax.scan`.

## Validation

Reference traces generated from Ludii's Java engine are stored in `tests/traces/`. The validation suite replays each trace through ludii-jax and compares board states, legal move counts, and termination at every step.

```bash
# Validate against Ludii reference traces
python tests/test_against_ludii.py

# Run compilation coverage on full corpus
python tests/test_coverage.py /path/to/ludii/games 200
```

## Acknowledgments

- [Ludii](https://ludii.games/) -- the game description language and corpus
- [JAX](https://github.com/jax-ml/jax) -- hardware-accelerated numerical computing
- [Lark](https://github.com/lark-parser/lark) -- parsing toolkit
- [PGX](https://github.com/sotetsuk/pgx) -- JAX game environments (inspiration for the API)

## License

MIT
