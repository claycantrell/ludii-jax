"""
Dynamic game state construction.

Only allocates the fields a specific game needs. The analysis pass
determines which fields are required, and this module builds the
namedtuple class and initial state.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

BOARD_DTYPE = jnp.int8
ACTION_DTYPE = jnp.int16
REWARD_DTYPE = jnp.float32
INDEX_DTYPE = jnp.int32   # for array indexing (Metal requires int32, not int8/int16)
EMPTY = -1  # sentinel value for empty cells (cast to BOARD_DTYPE in context)


def idx(x):
    """Cast to int32 for safe array indexing (Metal backend requires this)."""
    return jnp.asarray(x, dtype=jnp.int32)


class State:
    """Outer environment state wrapping the game-specific inner state."""
    __slots__ = ('game_state', 'current_player', 'legal_action_mask', 'winners',
                 'rewards', 'mover_reward', 'terminated', 'truncated', 'global_step_count')

    def __init__(self, game_state, current_player, legal_action_mask, winners, rewards,
                 mover_reward=None, terminated=None, truncated=None, global_step_count=None):
        self.game_state = game_state
        self.current_player = current_player
        self.legal_action_mask = legal_action_mask
        self.winners = winners
        self.rewards = rewards
        self.mover_reward = mover_reward if mover_reward is not None else REWARD_DTYPE(0.0)
        self.terminated = terminated if terminated is not None else jnp.bool_(False)
        self.truncated = truncated if truncated is not None else jnp.bool_(False)
        self.global_step_count = global_step_count if global_step_count is not None else ACTION_DTYPE(0)

    def replace(self, **kwargs):
        d = {s: getattr(self, s) for s in self.__slots__}
        d.update(kwargs)
        return State(**d)


# Register State as a JAX pytree
import jax
def _state_flatten(state):
    children = tuple(getattr(state, s) for s in State.__slots__)
    return children, None

def _state_unflatten(aux, children):
    return State(**dict(zip(State.__slots__, children)))

jax.tree_util.register_pytree_node(State, _state_flatten, _state_unflatten)


def build_game_state_class(info) -> tuple:
    """Build the GameState namedtuple class and initial values.

    Returns (GameStateCls, defaults_dict) where defaults_dict maps
    field names to their initial JAX arrays.
    """
    fields = [
        "board",
        "legal_action_mask",
        "current_player",
        "phase_idx",
        "phase_step_count",
        "previous_actions",
    ]
    defaults = {}

    n = info.topology.num_sites
    np = info.num_players
    # For stacking games, board layers = stack height instead of piece types
    if info.has_stacking:
        num_pieces = info.max_stack_height
    else:
        num_pieces = len(info.pieces) if info.pieces else 1

    # Always present
    defaults["board"] = jnp.ones((num_pieces, n), dtype=BOARD_DTYPE) * EMPTY
    defaults["current_player"] = BOARD_DTYPE(0)
    defaults["phase_idx"] = BOARD_DTYPE(0)
    defaults["phase_step_count"] = BOARD_DTYPE(0)
    defaults["previous_actions"] = -jnp.ones(np + 1, dtype=ACTION_DTYPE)

    # Conditional fields — only allocate what the compiled game actually reads/writes
    if info.has_score:
        fields.append("scores")
        defaults["scores"] = jnp.zeros(np, dtype=REWARD_DTYPE)

    if info.has_extra_turn:
        fields.append("extra_turn_fn_idx")
        defaults["extra_turn_fn_idx"] = ACTION_DTYPE(-1)

    if info.has_sow:
        fields.append("seed_counts")
        defaults["seed_counts"] = jnp.zeros(n, dtype=BOARD_DTYPE)
        fields.append("pit_owner")
        defaults["pit_owner"] = jnp.full(n, -1, dtype=BOARD_DTYPE)

    if info.has_dice:
        fields.append("dice_values")
        defaults["dice_values"] = jnp.zeros(info.dice_count, dtype=BOARD_DTYPE)

    # Stacking: not yet implemented (Connect Four, etc.)
    # if info.has_stacking:
    #     fields.append("stack_heights")
    #     defaults["stack_heights"] = jnp.zeros(n, dtype=BOARD_DTYPE)

    # Build namedtuple
    GameState = namedtuple("GameState", fields,
                           defaults=[defaults.get(f) for f in fields if f != "legal_action_mask"])

    return GameState, defaults
