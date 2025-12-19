# Model Predictive Control

A Python implementation of Model Predictive Control (MPC) for solving optimal control problems on dynamical systems.

## 📋 Table of Contents

- [Overview](#-overview)
- [Mathematical Background](#-mathematical-background)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Installation](#-installation)
- [Usage](#️-usage)
- [Code Components](#-code-components)
- [Visualization](#-visualization)
- [How It Works](#-how-it-works)

## 🎯 Overview

This project implements a **Model Predictive Control** framework that:

1. **Solves optimal control problems** for dynamical systems (quadrotor)
2. **Simulates and validates** controllers in various environments
3. **Visualizes trajectories** in real-time

MPC generates optimal control sequences by solving a constrained optimization problem at each time step.

## 📐 Mathematical Background

### Model Predictive Control

MPC solves a finite-horizon optimal control problem at each time step:
```
minimize    Σ L(x_t, u_t) + L_f(x_N)
  u_0,...,u_{N-1}

subject to  x_{t+1} = f(x_t, u_t)     (system dynamics)
            x_t ∈ X                    (state constraints)
            u_t ∈ U                    (control constraints)
```

Where:

* `x_t`: state at time t
* `u_t`: control input at time t
* `L(x,u)`: stage cost function
* `L_f(x)`: terminal cost function
* `f(x,u)`: system dynamics model
* `N`: prediction horizon

## 📁 Project Structure
```
high_mpc/
├── common/                    # Shared utilities
│   ├── logger.py             # Logging and data recording
│   ├── quad_index.py         # Quadrotor state/control indices
│   └── util.py               # Helper functions (seeding, GPU config)
│
├── mpc/                       # MPC solver implementation
│   ├── mpc.py                # MPC optimization and dynamics
│   └── saved/                # Compiled solver binaries
│       └── mpc_v1.so         # Cached optimization solver
│
├── simulation/                # Simulation environments
│   ├── dynamic_gap.py        # Quadrotor environment
│   └── animation.py          # Visualization
│
└── run_mpc.py                # Main entry point
```

## 🔧 Dependencies

This project requires:

* **Python 3.7+**
* **NumPy** - Numerical computing and array operations
* **SciPy** - Scientific computing and optimization
* **Matplotlib** - Visualization and animation
* **CasADi** - Symbolic framework for automatic differentiation and optimization

### Why These Dependencies?

* **CasADi**: Provides efficient symbolic differentiation for MPC optimization problems and interfaces with nonlinear programming solvers (IPOPT)
* **SciPy**: Alternative optimization tools and numerical integration
* **Matplotlib**: Creates visualizations and animations of control trajectories
* **NumPy**: Core numerical operations for state/control vectors and matrices

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MTheCreator/MPC.git
cd MPC
```

### 2. Install Dependencies

**Option A: Using pip**
```bash
pip install numpy scipy matplotlib casadi
```

**Option B: Using conda (recommended)**
```bash
conda create -n mpc python=3.8
conda activate mpc
conda install numpy scipy matplotlib
pip install casadi
```

### 3. Verify Installation
```bash
python -c "import casadi; import numpy; print('Dependencies OK')"
```

### 4. Generate Solver Binary (First Time Only)

The MPC solver needs to be compiled for your system:
```bash
cd high_mpc
```

Edit `mpc/mpc.py` and uncomment lines 248-250 (generation code), comment out lines 256-257 (loading code):
```python
# Uncomment these:
self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)
print("Generating shared library........")
cname = self.solver.generate_dependencies("mpc_v1.c")  
system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + self.so_path)

# Comment these out:
# print(self.so_path)
# self.solver = ca.nlpsol("solver", "ipopt", self.so_path, ipopt_options)
```

Then run once to generate:
```bash
python3 run_mpc.py
```

After generation, reverse the changes (comment generation, uncomment loading).

## ▶️ Usage

### Running MPC Controller

Navigate to the `high_mpc` directory and run:
```bash
cd high_mpc
python3 run_mpc.py
```

This will:

1. Initialize the simulation environment
2. Run the MPC controller
3. Display real-time animation of the quadrotor trajectory

### Configuration

Edit `run_mpc.py` to configure:

* MPC horizon length (`plan_T`)
* Sampling time step (`plan_dt`)
* Cost function weights (in `mpc/mpc.py`)

## 🧩 Code Components

### Core Utilities (`util.py`)
```python
set_global_seed(seed)           # Ensures reproducible results
get_dir(path)                   # Creates directories safely
```

**Logic:**

* Sets random seeds for Python and NumPy to ensure experiments are reproducible

### Index Definitions

* `quad_index.py` - Defines state indices for quadrotor (position, velocity, orientation)

**Purpose:** Named indices make code more readable than using raw numbers (e.g., `x[kPosX]` instead of `x[0]`)

### Logger (`logger.py`)

Tracks and saves:

* State trajectories
* Control inputs
* Cost evolution
* Computation times

### MPC Solver (`mpc/mpc.py`)

Core MPC implementation:

* System dynamics (quadrotor)
* Cost functions (tracking, control effort)
* Constraints (thrust limits, angular velocity bounds)
* Optimization using IPOPT solver

## 📊 Visualization

The animation displays:

* **Left panels**: Time-series plots of position, velocity, attitude, and control inputs
* **Right panel**: 3D visualization of quadrotor trajectory with predicted path

## 🔬 How It Works

1. **MPC Loop:**
   * At each time step, solve optimization problem for next N steps
   * Apply first control input to the system
   * Measure new state
   * Repeat with updated state

2. **Optimization:**
   * Uses CasADi for symbolic differentiation
   * IPOPT solver finds optimal control sequence
   * Compiled `.so` file accelerates repeated solves

3. **Visualization:**
   * Matplotlib animation updates in real-time
   * Shows both actual trajectory and MPC predictions