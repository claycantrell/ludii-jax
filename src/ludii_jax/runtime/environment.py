"""
LudiiJAX Environment: gym-like API for compiled Ludii games.

env = Environment(game_rules, game_info)
state = env.init(rng)
state = env.step(state, action, key)
"""

from typing import Optional, Any

import jax
import jax.numpy as jnp

from .state import State, BOARD_DTYPE, ACTION_DTYPE, REWARD_DTYPE, EMPTY

MAX_STEP_COUNT = 2000


def _select_state(cond, true_state, false_state):
    """Branchless state selection (works on Metal/GPU without lax.cond)."""
    return jax.tree.map(lambda t, f: jnp.where(cond, t, f), true_state, false_state)


class Environment:
    """JAX-accelerated game environment compiled from a Ludii .lud file."""

    def __init__(self, game_rules: dict, game_state_cls, game_info):
        self.game_info = game_info
        self.game_state_cls = game_state_cls
        self.num_sites = game_info.topology.num_sites
        self.num_players = game_info.num_players
        if game_info.has_stacking:
            self.num_pieces = game_info.max_stack_height
        else:
            self.num_pieces = len(game_info.pieces) if game_info.pieces else 1

        self.num_actions = game_rules['action_size']
        self._start = game_rules['start_rules']
        self._apply_action = game_rules['apply_action_fn']
        self._get_legal = game_rules['legal_action_mask_fn']
        self._apply_effects = game_rules['apply_effects_fn']
        self._get_next_player = game_rules['next_player_fn']
        self._get_winner = game_rules['end_rules']
        self._update_info = game_rules.get('addl_info_fn', lambda s, a: s)

    def init(self, rng) -> State:
        game_state = self.game_state_cls(
            board=jnp.ones((self.num_pieces, self.num_sites), dtype=BOARD_DTYPE) * EMPTY,
            legal_action_mask=jnp.ones(self.num_actions, dtype=jnp.bool_),
            current_player=BOARD_DTYPE(0),
            phase_idx=BOARD_DTYPE(0),
            phase_step_count=BOARD_DTYPE(0),
            previous_actions=-jnp.ones(self.num_players + 1, dtype=ACTION_DTYPE),
        )

        game_state = self._start(game_state)
        current_player = self._get_next_player(game_state)
        game_state = game_state._replace(current_player=current_player)
        game_state = self._update_info(game_state, -1)

        legal = self._get_legal(game_state).astype(jnp.bool_)
        game_state = game_state._replace(legal_action_mask=legal)

        return State(
            game_state=game_state,
            legal_action_mask=legal,
            current_player=current_player,
            winners=EMPTY * jnp.ones(self.num_players, BOARD_DTYPE),
            rewards=jnp.zeros(self.num_players, dtype=REWARD_DTYPE),
        )

    def step(self, state: State, action, key=None) -> State:
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player
        already_done = state.terminated | state.truncated

        # Always compute the step (branchless for Metal/GPU compatibility)
        stepped = self._step(state, action, key)

        # If already terminated, return zeros rewards instead of step result
        state = _select_state(already_done, state.replace(rewards=jnp.zeros_like(state.rewards)), stepped)

        # If illegal action, apply penalty
        illegal_state = self._illegal_action(state, current_player)
        state = _select_state(is_illegal, illegal_state, state)

        # If terminated, set all actions legal (for API consistency)
        done_state = state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask))
        state = _select_state(state.terminated, done_state, state)

        return state

    def _step(self, state: State, action, key) -> State:
        gs = state.game_state

        # Roll dice if needed
        if hasattr(gs, 'dice_values') and key is not None:
            num_dice = gs.dice_values.shape[0]
            dice = jax.random.randint(key, (num_dice,), 1, 7).astype(BOARD_DTYPE)
            gs = gs._replace(dice_values=dice)

        original_player = gs.current_player

        # Apply action
        gs = self._apply_action(gs, action)

        # Apply effects
        gs = self._apply_effects(gs, original_player)

        # Update phase step count
        gs = gs._replace(phase_step_count=gs.phase_step_count + 1)

        # Get next player
        next_player = self._get_next_player(gs)
        gs = gs._replace(current_player=next_player)

        # Update additional info
        gs = self._update_info(gs, action)

        # Compute legal actions for next player
        legal = self._get_legal(gs).astype(jnp.bool_)
        gs = gs._replace(legal_action_mask=legal)

        # Check end conditions
        winners, terminated = self._get_winner(gs._replace(current_player=original_player))
        terminal_rewards = jnp.where(
            (winners == EMPTY).all(),
            jnp.zeros_like(winners),
            jnp.where(winners, 1, -1)
        ).astype(REWARD_DTYPE)
        rewards = jax.lax.select(terminated, terminal_rewards, jnp.zeros_like(terminal_rewards))

        # Check truncation
        step_count = state.global_step_count + 1
        truncated = step_count >= MAX_STEP_COUNT

        return state.replace(
            game_state=gs,
            current_player=next_player,
            legal_action_mask=legal,
            winners=winners,
            rewards=rewards,
            mover_reward=rewards[original_player.astype(jnp.int32)],
            terminated=terminated,
            truncated=truncated,
            global_step_count=step_count,
        )

    def _illegal_action(self, state, player):
        penalty = REWARD_DTYPE(1.0)
        reward = jnp.ones_like(state.rewards) * (-penalty) * (self.num_players - 1)
        reward = reward.at[player].set(penalty * (self.num_players - 1))
        return state.replace(rewards=reward, terminated=jnp.bool_(True))
