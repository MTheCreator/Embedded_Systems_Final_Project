# Embedded Systems Final Project - Quadrotor Control

Multiple control strategies for quadrotor UAVs: Model Predictive Control (MPC), Reinforcement Learning (DDPG), and H∞ robust control for tilt-rotor systems.

## Project Structure

```
├── MPC/                    # Model Predictive Control
├── RL/               # Reinforcement Learning that contains the DDPG algorithm and the LTL specifications but only mind the DDPG algorithm since the LTL specifications will be explained in the other repository (LTL)
├── Paper_Visualization/   # Tilt-rotor UAV H∞ control
├── Paper_Implementation/  # Research paper & notebook
├── project_demos/         # Demo videos and GIFs
└── General_Report.pdf     # Full project report
```

## Quick Start

**MPC:**
```bash
cd MPC/high_mpc
python3 run_mpc.py
```

**RL:**
```bash
cd RL/RL_DDPG
python main_no_pybullet.py
```

**Tilt-Rotor:**
```bash
cd Paper_Visualization
python run_pybullet_visual.py
```

## Components

### MPC
Optimal control using constrained optimization with CasADi and IPOPT solver.
[Documentation](MPC/README.md)

### RL (DDPG)
Learning-based control with obstacle avoidance using Deep Deterministic Policy Gradient.
[Documentation](RL/RL_DDPG/README.md)

### Tilt-Rotor Control
H∞ robust control implementation based on research paper "Nonlinear optimal control for UAVs with tilting rotors".
[Documentation](Paper_Visualization/README.md)

## Demos
![Paper Control](project_demos/demo_speed.gif)
![Webot Simulation](project_demos/UAV.gif)
![RL Hovering](project_demos/RL_Hovering_no_obstacle.gif)
![MPC Simulation](project_demos/MPC_gif_drone.gif)

[View All Demos](project_demos/README.md)

## Documentation

- **Full Report:** [General_Report.pdf](General_Report.pdf)
- **Research Paper:** [Paper.pdf](Paper_Visualization/Nonlinear_optimal_control_for_UAVs_with_tilting_rotors.pdf)
- **Interactive Notebook:** [Paper_Implementation.ipynb](Paper_Implementation/Paper_Implementation.ipynb)

## Dependencies

```bash
# MPC
pip install numpy scipy matplotlib casadi

# RL
pip install torch pygame

# Tilt-Rotor
pip install pybullet jupyter
```
