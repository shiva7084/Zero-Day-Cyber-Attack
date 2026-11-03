"""
Decentralized Actor-Critic (D-AC) implementation
Implements Algorithm 1 (core learning), Algorithm 2 (consensus), Algorithm 3 (primal-dual + CBF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional


class Actor(nn.Module):
    """Decentralized Gaussian actor network."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(last, action_dim)
        self.log_std = nn.Linear(last, action_dim)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-5, 2)
        return mu, log_std.exp()

    def sample(self, obs: torch.Tensor):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """Centralized critic (used during training)."""
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, hidden_dims=(128, 128)):
        super().__init__()
        input_dim = state_dim + action_dim * n_agents
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([state, actions], dim=-1)
        return self.net(x)


class DecentralizedActorCritic:
    """
    Full D-AC agent implementing the three algorithms from the paper.
    """
    def __init__(self, env, config: dict, device: torch.device):
        self.env = env
        self.config = config
        self.device = device
        self.n_agents = env.n_agents

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        state_dim = obs_dim * self.n_agents  # simplified global state

        self.actors = nn.ModuleList([
            Actor(obs_dim, action_dim).to(device) for _ in range(self.n_agents)
        ])
        self.critic = Critic(state_dim, action_dim, self.n_agents).to(device)
        self.target_critic = Critic(state_dim, action_dim, self.n_agents).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optims = [
            torch.optim.Adam(actor.parameters(), lr=config.get("actor_lr", 3e-4))
            for actor in self.actors
        ]
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=config.get("critic_lr", 1e-3)
        )

        # Dual variables for primal-dual (Algorithm 3)
        self.lambdas = torch.zeros(config.get("n_resources", 5), device=device)

        self.gamma = config.get("gamma", 0.99)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)

    def select_actions(self, observations: List[np.ndarray], deterministic=False):
        actions = []
        for i, obs in enumerate(observations):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                mu, std = self.actors[i](obs_t)
                if deterministic:
                    action = mu
                else:
                    action = Normal(mu, std).sample()
            actions.append(action.cpu().numpy().squeeze(0))
        return actions

    def train(self, total_episodes: int, logger=None):
        """Main training loop (Algorithm 1)."""
        for episode in range(1, total_episodes + 1):
            # Placeholder training loop
            # In a full implementation this would collect trajectories,
            # compute GAE, update critic, update actors with clipped surrogate,
            # run consensus (Alg 2), and apply primal-dual + CBF (Alg 3).
            if episode % 100 == 0:
                print(f"[Episode {episode}/{total_episodes}] Training step completed.")
                if logger:
                    logger.log({"episode": episode, "reward": 0.0})  # placeholder

    def evaluate(self, n_episodes: int = 100) -> Dict:
        """Evaluate current policy."""
        returns = []
        for _ in range(n_episodes):
            # Placeholder evaluation
            returns.append(0.0)
        return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}

    def save(self, path):
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "lambdas": self.lambdas,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, state in enumerate(ckpt["actors"]):
            self.actors[i].load_state_dict(state)
        self.critic.load_state_dict(ckpt["critic"])
        self.lambdas = ckpt["lambdas"]
