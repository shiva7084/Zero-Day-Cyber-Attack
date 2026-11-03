# Decentralized Actor-Critic Architectures for Resource Allocation in Multi-Agent Cyber-Physical Systems

**Official code and data release** accompanying the paper:

> Mallu Shiva Rama Krishna, Mulkala Saritha, Kiran Dasari, K. Anuradha  
> *Decentralized Actor-Critic Architectures for Resource Allocation in Multi-Agent Cyber-Physical Systems*  
> Inteligencia Artificial, 2026 (or current volume)

## Overview

This repository implements the **Decentralized Actor-Critic (D-AC)** framework proposed in the paper. The architecture combines:

- Decentralized actor networks
- Critic-assisted cooperative optimization
- Graph-based distributed consensus
- Adaptive primal-dual resource management
- Control Barrier Function (CBF) based safety enforcement

## Repository Structure

```
Decentralized-Actor-Critic-CPS/
├── src/                    # Core source code
│   ├── agents/             # Actor & Critic networks
│   ├── algorithms/         # D-AC Algorithms 1, 2, 3
│   ├── envs/               # Multi-agent CPS environments
│   ├── utils/              # Consensus, primal-dual, CBF utilities
│   └── train.py            # Main training script
├── configs/                # Experiment configurations
├── data/                   # Datasets for different scenarios
│   ├── static_workload/
│   ├── dynamic_workload/
│   ├── energy_aware/
│   ├── communication_constrained/
│   ├── heterogeneous_agents/
│   ├── scalability/
│   └── disturbance/
├── notebooks/              # Analysis & visualization notebooks
├── results/                # Logged results & figures
├── docs/                   # Additional documentation
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/ShivaRamaKrishna-Mallu/Decentralized-Actor-Critic-CPS.git
cd Decentralized-Actor-Critic-CPS
pip install -r requirements.txt
```

## Quick Start

```bash
# Train D-AC on the default (Medium Network) configuration
python src/train.py --config configs/medium_network.yaml

# Evaluate a trained model
python src/train.py --config configs/medium_network.yaml --eval --checkpoint results/best_model.pt
```

## Experimental Scenarios

The code supports all scenarios reported in the paper:

| Scenario                  | Description                                      |
|---------------------------|--------------------------------------------------|
| Static Workload           | Constant task arrivals                           |
| Dynamic Workload          | Variable arrivals + deadlines                    |
| Energy-Aware              | Battery & energy constraints                     |
| Communication-Constrained | Limited bandwidth + packet loss                  |
| Heterogeneous Agents      | Different capacities per agent                   |
| Scalability (5–80 agents) | Scaling study                                    |
| Disturbance               | Agent failures + stochastic noise                |

## Citation

If you use this code or data, please cite:

```bibtex
@article{krishna2026dac,
  title   = {Decentralized Actor-Critic Architectures for Resource Allocation in Multi-Agent Cyber-Physical Systems},
  author  = {Krishna, Mallu Shiva Rama and Saritha, Mulkala and Dasari, Kiran and Anuradha, K.},
  journal = {Inteligencia Artificial},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Corresponding author: Kiran Dasari (kiran.dasari@manipal.edu)  
For code-related questions: Mallu Shiva Rama Krishna
