# Quadcopter RL with LTL Specifications

Deep Deterministic Policy Gradient (DDPG) agent learning to fly a quadcopter with formal Linear Temporal Logic (LTL) specifications for safety and task completion.

## What Got Fixed

### Critical Physics Bugs

1. **Rotation Matrix** - The original code had completely broken thrust transformation:
   ```python
   # WRONG (original):
   thrust_world = np.array([
       total_thrust * np.sin(roll) * np.cos(yaw),
       total_thrust * np.sin(pitch) * np.sin(yaw),
       total_thrust * np.cos(roll) * np.cos(pitch)
   ])
   
   # CORRECT (fixed):
   R = rotation_matrix(roll, pitch, yaw)  # proper ZYX Euler
   thrust_world = R @ thrust_body
   ```
   The original mixed up axes and didn't use proper rotation math. Now uses correct 3D rotation matrices.

2. **Inertia Tensor** - Changed from scalar to proper 3x3 matrix:
   ```python
   # WRONG:
   self.inertia = 0.01
   
   # CORRECT:
   self.inertia = np.array([
       [0.01, 0, 0],   # Ixx (roll)
       [0, 0.01, 0],   # Iyy (pitch)  
       [0, 0, 0.02]    # Izz (yaw, higher)
   ])
   ```
   Real quadcopters have different inertias around different axes.

3. **Drag Force** - Fixed the weird element-wise abs() issue:
   ```python
   # WRONG:
   drag_force = -self.linear_drag * self.velocity * np.abs(self.velocity)
   
   # CORRECT:
   v_mag = np.linalg.norm(self.velocity)
   if v_mag > 0:
       drag_direction = -self.velocity / v_mag
       drag_force = drag_direction * drag_coeff * v_mag * v_mag
   ```

4. **LTL Circular Reference** - Fixed atomic propositions evaluating each other multiple times per timestep:
   ```python
   # Added caching to avoid re-evaluation:
   if env.current_step == self._cached_step:
       return self._cached_value
   ```

5. **Hover Bias** - Made it optional and properly decaying:
   ```python
   # Now it's a training option that can be disabled
   use_hover_bias=False  # Can enable for easier initial learning
   ```

### Motor Layout

Motors are arranged in X-configuration:
```
    1(FL)  0(FR)
       \ /
       / \
    2(BL)  3(BR)
```

- Motors 0, 2: Clockwise (CW)
- Motors 1, 3: Counter-clockwise (CCW)

Roll = left/right tilt, Pitch = forward/back tilt, Yaw = rotation

## What It Does

### DDPG Agent
- **Actor network**: Maps state → motor thrust commands (continuous actions)
- **Critic network**: Estimates Q-values for state-action pairs
- **Target networks**: Stabilize training with soft updates
- **Experience replay**: Breaks temporal correlations
- **Exploration noise**: Gaussian noise that decays over time

### Environment
- Simulates quadcopter physics at 50 Hz
- 3D position, velocity, orientation (Euler angles), angular velocity
- Obstacles (spheres and boxes) for collision avoidance
- Target position for hovering task
- Pygame visualization with multiple views

### LTL Specifications
Formal temporal logic properties that shape rewards:

**Safety (□¬bad)** - must never happen:
- No collisions with obstacles
- Stay in bounds
- Don't flip over

**Liveness (◊good)** - must eventually happen:
- Reach target zone
- Achieve stable flight

**Response (□(A → ◊B))** - if-then rules:
- If near obstacle → move away
- If near target → stabilize

**Persistence (◊□prop)** - maintain once achieved:
- Once hovering stably → keep hovering

## Installation

```bash
pip install torch numpy pygame
```

That's it. No other dependencies.

## Usage

### Training
```python
# In main_ltl.py, set:
MODE = "train"
EPISODES = 3000
NUM_OBSTACLES = 2
USE_HOVER_BIAS = False  # True for easier initial training

python main_ltl.py
```

The agent will train and save:
- `best_drone_ltl.pth` - best model so far
- `drone_ltl_ep{N}.pth` - checkpoints every 100 episodes
- `drone_ltl_interrupted.pth` - if you Ctrl+C

### Testing
```python
# In main_ltl.py, set:
MODE = "test"
MODEL_PATH = "best_drone_ltl.pth"

python main_ltl.py
```

Runs episodes with no exploration noise, shows LTL compliance stats.

## State Space (27 dimensions)

```
Position (3):        [x, y, z] / max_pos
Orientation (3):     [roll, pitch, yaw] / π
Velocity (3):        [vx, vy, vz] / max_vel
Angular vel (3):     [ωx, ωy, ωz] / 10.0
Target rel pos (3):  (target - pos) / max_pos
Obstacles (12):      3 nearest obstacles × (rel_pos(3) + dist(1))
```

Everything normalized to roughly [-1, 1] for neural network training.

## Action Space (4 dimensions)

Motor thrust commands in [0, 1], scaled to [0, max_thrust_per_motor].

## Reward Structure

**Base rewards** (from environment):
- Progress toward target: +25 × distance_improvement
- Reaching target zone: +15
- Stable at target: +10 (velocity), +10 (level), +8 (stable)
- Perfect hover: +25
- Success (hover 100 steps): +500
- Collisions/crashes: -20 to -100
- Alive bonus: +0.1 per step

**LTL rewards** (from specifications):
- Safety violations: -80 to -200
- Liveness achievements: +30 to +50
- Response satisfactions: +15 to +25
- Persistence achievement: +300

Total reward = base + LTL

## Hyperparameters

```python
# DDPG
learning_rate_actor = 1e-4
learning_rate_critic = 1e-3
gamma = 0.99  # discount factor
tau = 0.005   # soft update rate
batch_size = 128
buffer_size = 100000

# Exploration
noise_std_initial = 0.3
noise_std_min = 0.05
noise_decay = 0.99995

# Physics
dt = 0.02  # 50 Hz
mass = 1.0 kg
max_thrust_per_motor = 4.0 N
arm_length = 0.25 m
```

## Known Limitations

1. **Euler angles** - Still uses roll/pitch/yaw instead of quaternions. This means:
   - Gimbal lock possible at ±90° pitch
   - Discontinuities at ±180°
   - Should upgrade to quaternions for robustness

2. **Simplified aerodynamics**:
   - Linear drag model (real quads have complex blade/body interactions)
   - No ground effect
   - No prop wash or rotor dynamics
   - No wind or turbulence

3. **Thrust model**:
   - Assumes instant thrust response (real motors have lag)
   - No battery dynamics
   - Simplified yaw torque (real quads use prop speed differences)

4. **LTL monitor overhead**:
   - Evaluates all specs every timestep
   - Could optimize for large spec sets

## File Structure

```
ddpg_agent.py          - DDPG implementation with actor/critic networks
env_enhanced_ltl.py    - Quadcopter physics simulation + LTL integration
ltl_specifications.py  - LTL formal spec definitions and monitoring
main_ltl.py            - Training/testing script
```

## Tips for Training

1. **Start simple**: Train with `NUM_OBSTACLES = 0` first to learn basic hovering
2. **Use hover bias**: Set `USE_HOVER_BIAS = True` for faster initial learning
3. **Monitor LTL stats**: Check which specs are being violated most
4. **Adjust reward weights**: If agent ignores certain specs, increase their penalties
5. **Be patient**: Good hovering behavior takes 500-1000 episodes
6. **Save often**: Training can be unstable, checkpoints are your friend

## Future Improvements

- [ ] Quaternion orientation (eliminate gimbal lock)
- [ ] More realistic thrust dynamics (motor lag, prop physics)
- [ ] Domain randomization (varying mass, drag, etc.)
- [ ] Curriculum learning (gradually add obstacles)
- [ ] Multi-waypoint missions
- [ ] Vision-based control (instead of perfect state)
- [ ] Transfer to real hardware (sim-to-real gap)
- [ ] Multi-agent scenarios (formation flying)

## References

**DDPG**:
- Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)

**LTL for RL**:
- Hasanbeig et al., "Deep Reinforcement Learning with Temporal Logics" (2018)
- Li et al., "Reinforcement Learning with Temporal Logic Rewards" (2017)

**Quadcopter Dynamics**:
- Bouabdallah, "Design and control of quadrotors with application to autonomous flying" (2007)

## License

Do whatever you want with this code. No warranties. Don't blame me if your drone crashes.

---

**Note**: This is a simulation. Real quadcopters are way harder. They have:
- Sensor noise (IMU drift, GPS errors)
- Wind disturbances
- Battery voltage drop affecting thrust
- Communication delays
- Structural flexibility
- And a million other things that will ruin your day

But hey, at least in simulation we can undo the crashes with Ctrl+R.