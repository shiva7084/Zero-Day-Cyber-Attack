#!/usr/bin/env python3
"""
Main training script for Decentralized Actor-Critic (D-AC)
for resource allocation in Multi-Agent Cyber-Physical Systems.

Implements Algorithms 1, 2 and 3 from the paper.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from algorithms.dac import DecentralizedActorCritic
from envs.cps_env import MultiAgentCPSEnv
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train D-AC for Multi-Agent CPS Resource Allocation")
    parser.add_argument("--config", type=str, default="configs/medium_network.yaml",
                        help="Path to experiment configuration file")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = MultiAgentCPSEnv(
        n_agents=config["n_agents"],
        scenario=config["scenario"],
        max_steps=config.get("max_steps", 200),
    )

    # Initialize D-AC agent
    agent = DecentralizedActorCritic(
        env=env,
        config=config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    logger = Logger(log_dir=config.get("log_dir", "results"))

    if args.eval:
        assert args.checkpoint is not None, "Checkpoint required for evaluation"
        agent.load(args.checkpoint)
        metrics = agent.evaluate(n_episodes=config.get("eval_episodes", 100))
        print("Evaluation metrics:", metrics)
    else:
        agent.train(
            total_episodes=config.get("total_episodes", 3000),
            logger=logger,
        )
        agent.save(Path(config.get("log_dir", "results")) / "best_model.pt")

    print("Done.")


if __name__ == "__main__":
    main()
