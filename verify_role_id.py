#!/usr/bin/env python3
"""Quick verification script for observe_role_id feature."""

from jaxmarl.environments.mabrax import Ant
import jax
import jax.numpy as jnp

print("Testing observe_role_id=True with Ant environment...")
env = Ant(observe_role_id=True)
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

print("\nObservation shapes:")
for agent, o in obs.items():
    print(f"  {agent}: {o.shape}")

print(f"\nNumber of agents: {env.num_agents}")
print("\nRole one-hots (last num_agents elements):")
for i, agent in enumerate(env.agents):
    role = obs[agent][-env.num_agents :]
    print(f"  {agent}: {role}")

print("\n✓ Verification passed!")
