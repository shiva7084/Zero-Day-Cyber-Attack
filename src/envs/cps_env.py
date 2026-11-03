"""
Multi-Agent Cyber-Physical System Environment
Supports all experimental scenarios described in the paper.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Tuple, Optional


class MultiAgentCPSEnv(gym.Env):
    """
    Simplified multi-agent CPS environment for resource allocation.
    State of each agent: [Energy, Compute, Bandwidth, Queue, Demand, Latency]
    Action of each agent: [compute_alloc, bandwidth_alloc, energy, power, tau]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_agents: int = 10,
        scenario: str = "static_workload",
        max_steps: int = 200,
        resource_capacity: Optional[Dict] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.scenario = scenario
        self.max_steps = max_steps
        self.current_step = 0

        # Local observation / action dimensions (from paper)
        self.obs_dim = 6   # E, C, B, Q, D, L
        self.act_dim = 5   # c, b, e, p, tau

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.act_dim,), dtype=np.float32
        )

        self.resource_capacity = resource_capacity or {
            "compute": 100.0,
            "bandwidth": 50.0,
            "energy": 200.0,
        }

        self._reset_agents()

    def _reset_agents(self):
        self.agent_states = np.random.uniform(0.3, 0.9, size=(self.n_agents, self.obs_dim)).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._reset_agents()
        observations = [self.agent_states[i].copy() for i in range(self.n_agents)]
        return observations, {}

    def step(self, actions: List[np.ndarray]):
        self.current_step += 1
        actions = np.array(actions)

        # Simple dynamics placeholder
        reward = 0.0
        for i in range(self.n_agents):
            # Resource utilization contribution
            reward += float(np.mean(actions[i])) * 0.5
            # Update local state (very simplified)
            self.agent_states[i, 0] = np.clip(self.agent_states[i, 0] - 0.01 * actions[i, 2], 0.05, 1.0)  # energy
            self.agent_states[i, 3] = np.clip(self.agent_states[i, 3] + 0.05 - 0.1 * actions[i, 0], 0.0, 1.0)  # queue

        # Shared constraint penalty (simplified)
        total_compute = np.sum(actions[:, 0])
        if total_compute > 1.0:
            reward -= (total_compute - 1.0) * 2.0

        terminated = self.current_step >= self.max_steps
        truncated = False

        observations = [self.agent_states[i].copy() for i in range(self.n_agents)]
        infos = [{"agent_id": i} for i in range(self.n_agents)]

        return observations, reward, terminated, truncated, infos

    def render(self):
        print(f"Step {self.current_step} | Mean energy: {self.agent_states[:, 0].mean():.3f}")
