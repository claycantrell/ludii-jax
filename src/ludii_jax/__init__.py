"""
ludii_jax: Compile Ludii .lud board games to JAX-accelerated environments.

    from ludii_jax import compile
    env = compile("games/tic_tac_toe.lud")
    state = env.init(jax.random.PRNGKey(0))
    state = env.step(state, action)
"""

from .compile import compile

__all__ = ["compile"]
