# Tiltrotor UAV Simulation Framework

A modular simulation framework for tilt-rotor quadrotor UAVs with H∞ robust control, Kalman filtering, and PyBullet physics integration, inspired from the paper Nonlinear optimal control for UAVs with tilting rotors.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Tilt-rotor dynamics**: Full nonlinear 12-state model with independently tilting rotors
- **H∞ robust control**: Advanced controller with guaranteed disturbance rejection
- **State estimation**: H∞ Kalman filter for sensor fusion
- **Autonomous navigation**: Waypoint following with obstacle avoidance
- **Dual physics engines**: Simple Euler integrator + PyBullet integration
- **3D visualization**: Real-time PyBullet rendering with camera tracking

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run PyBullet 3D Visualization 

```bash
python run_pybullet_visual.py
```

Watch the drone fly in 3D with:
- Auto-following camera
- Real-time trajectory display
- Correct waypoint navigation
- Obstacle avoidance

### Run Simple Simulation

```bash
python run_simulation.py
```

### Interactive Notebook

```bash
jupyter notebook demo.ipynb
```

## Project Structure

```
tiltrotor_sim/
├── models/         # UAV dynamics, parameters, linearization
├── control/        # H∞ controller, Kalman filter
├── navigation/     # Waypoint guidance, obstacles
├── physics/        # Simple & PyBullet simulators
├── visualization/  # Plotting utilities
└── utils/          # Simulation orchestrator
```

## Usage Examples

### Basic Simulation

```python
import numpy as np
from tiltrotor_sim.models import TiltRotorUAVParameters, TiltRotorUAV
from tiltrotor_sim.navigation import SphereObstacle
from tiltrotor_sim.utils import SimulationRunner

# Setup
params = TiltRotorUAVParameters()
uav = TiltRotorUAV(params)

waypoints = [
    np.array([5.0, 5.0, 5.0]),
    np.array([10.0, 10.0, 5.0])
]

obstacles = [
    SphereObstacle(center=np.array([7.5, 7.5, 3.0]), radius=1.0)
]

# Run simulation
runner = SimulationRunner(uav, waypoints, obstacles)
x0 = np.zeros(12)
x0[4] = 1.0  # Start at z=1m

results = runner.run(x0=x0, t_final=50.0)
```

### PyBullet Visualization

```python
# Uses simple physics (correct behavior) with PyBullet visualization
runner = SimulationRunner(
    uav, waypoints, obstacles,
    use_pybullet=False  # Simple physics
)

# Run and visualize
results = runner.run(x0, t_final=30.0)

# Plot results
from tiltrotor_sim.visualization import SimulationPlotter
SimulationPlotter.plot_3d_trajectory(
    results[1], waypoints=waypoints, obstacles=obstacles
)
```

## Configuration

### Controller Tuning

```python
runner = SimulationRunner(
    uav, waypoints, obstacles,
    controller_params={
        'r': 5.0,      # Control effort weight
        'rho': 0.5     # Disturbance attenuation
    }
)
```

### State Weighting

```python
Q = np.diag([
    100, 150,   # X position and velocity
    100, 150,   # Y position and velocity
    200, 300,   # Z position and velocity
    30, 5,      # Roll and roll rate
    30, 5,      # Pitch and pitch rate
    20, 2       # Yaw and yaw rate
])

results = runner.run(x0, t_final=100.0, Q_matrix=Q)
```

## System Model

**State vector (12 states)**:
```
x = [x, ẋ, y, ẏ, z, ż, φ, φ̇, θ, θ̇, ψ, ψ̇]ᵀ
```

**Control vector (6 inputs)**:
```
u = [θ₁, θ₂, θ₃, θ₄, τ₁, τ₂]ᵀ
```

where θᵢ are rotor tilt angles and τ₁, τ₂ are auxiliary torques.



## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- PyBullet (for 3D visualization)
- Jupyter (for notebooks)



## Contributing

Contributions welcome! Areas for improvement:
- Additional control strategies
- Advanced path planning
- Sensor noise models
- Wind disturbance models

---

**Get Started**: `python run_pybullet_visual.py` 
