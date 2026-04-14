"""
Compose compiled functions into a complete game step function.

Takes move functions, effect functions, end conditions, and phase
transitions and assembles them into the init/step/legal API.
"""

import jax
import jax.numpy as jnp

from ..runtime.state import BOARD_DTYPE, ACTION_DTYPE, REWARD_DTYPE, EMPTY


def compose_game(
    action_size: int,
    legal_fn,
    apply_fn,
    effects_fn,
    end_fn,
    next_player_fn,
    start_fn=None,
    num_players: int = 2,
):
    """Assemble a complete game rules dictionary.

    Returns dict with keys matching what Environment expects:
    action_size, apply_action_fn, legal_action_mask_fn,
    apply_effects_fn, start_rules, end_rules, next_player_fn,
    next_phase_fn, addl_info_fn.
    """
    return {
        'action_size': action_size,
        'apply_action_fn': apply_fn,
        'legal_action_mask_fn': legal_fn,
        'apply_effects_fn': effects_fn,
        'start_rules': start_fn or (lambda state: state),
        'end_rules': end_fn,
        'next_player_fn': next_player_fn,
        'next_phase_fn': lambda state: BOARD_DTYPE(0),
        'addl_info_fn': lambda state, action: state,
    }


def make_alternating_player_fn(num_players: int):
    """Simple alternating turn order: P1, P2, P1, P2, ..."""
    def next_player_fn(state):
        return (state.phase_step_count % num_players).astype(BOARD_DTYPE)
    return next_player_fn


def make_multi_phase(phase_dicts: list, num_players: int):
    """Combine multiple phase dicts into one with phase switching.

    Each phase_dict has: action_size, legal_fn, apply_fn, effects_fn.
    Pads action spaces to max size.
    """
    if len(phase_dicts) == 1:
        return phase_dicts[0]

    max_action_size = max(d['action_size'] for d in phase_dicts)

    # Pad each phase
    padded_legal = []
    padded_apply = []
    for d in phase_dicts:
        if d['action_size'] < max_action_size:
            pad = max_action_size - d['action_size']
            orig_legal = d['legal_fn']
            def make_padded(fn=orig_legal, p=pad):
                def padded(state):
                    base = fn(state)
                    return jnp.concatenate([base, jnp.zeros(p, dtype=BOARD_DTYPE)])
                return padded
            padded_legal.append(make_padded())

            orig_apply = d['apply_fn']
            def make_padded_apply(fn=orig_apply, s=d['action_size']):
                def padded(state, action):
                    return fn(state, jnp.minimum(action, s - 1))
                return padded
            padded_apply.append(make_padded_apply())
        else:
            padded_legal.append(d['legal_fn'])
            padded_apply.append(d['apply_fn'])

    def legal_fn(state):
        return jax.lax.switch(state.phase_idx, padded_legal, state)

    def apply_fn(state, action):
        return jax.lax.switch(state.phase_idx, padded_apply, state, action)

    return {
        'action_size': max_action_size,
        'legal_fn': legal_fn,
        'apply_fn': apply_fn,
    }
