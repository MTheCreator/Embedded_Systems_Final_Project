# 🚁 Drone RL - Obstacle Navigation

A reinforcement learning project using **DDPG** to train a drone to navigate obstacles and maintain stable hovering.

## Quick Start

### Install Dependencies
```bash
pip install numpy torch pygame
```

### Run the Project
```bash
# Training mode
python main_no_pybullet.py
```

Then change `MODE = "train"` in the file

## Features
- **3D drone physics** with realistic dynamics  
- **Obstacle avoidance** with spheres and boxes  
- **Stable hovering** requirement at target  
- **PyGame visualization** with multiple views  
- **DDPG algorithm** for continuous control  

## Files
- `ddpg_agent.py` - DDPG reinforcement learning agent  
- `env_enhanced.py` - Drone environment with physics  
- `main_no_pybullet.py` - Main training/testing script  

## Configuration
Edit `main_no_pybullet.py`:
- `MODE`: "train" or "test"  
- `EPISODES`: Training episodes  
- `NUM_OBSTACLES`: Number of obstacles  

## Goal
Train the drone to:
1. Navigate around obstacles  
2. Reach the target position  
3. Hover stably for 2 seconds  

## Requirements
- Python 3.8+  
- PyTorch  
- NumPy  
- PyGame