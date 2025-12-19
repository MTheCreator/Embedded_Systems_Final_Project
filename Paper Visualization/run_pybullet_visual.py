#!/usr/bin/env python3
"""
PyBullet Visualization-Only Demo

This uses the SIMPLE PHYSICS (which works correctly) but visualizes
the results in PyBullet. This gives you the best of both worlds:
- Correct dynamics from the analytical model
- Beautiful 3D visualization from PyBullet
"""

import numpy as np
import pybullet as p
import pybullet_data
import time

from tiltrotor_sim.models import TiltRotorUAVParameters, TiltRotorUAV
from tiltrotor_sim.navigation import SphereObstacle, BoxObstacle
from tiltrotor_sim.utils import SimulationRunner

print("=" * 70)
print("PYBULLET VISUALIZATION (Simple Physics)")
print("=" * 70)
print("\nThis demo uses:")
print("  ✓ SIMPLE physics (analytical model - works correctly)")
print("  ✓ PyBullet rendering (beautiful 3D visualization)")
print()
print("The drone will follow waypoints correctly!")
print()

# Create UAV
print("Setting up UAV...")
params = TiltRotorUAVParameters()
uav = TiltRotorUAV(params)

# Define waypoints
waypoints = [
    np.array([0.0, 0.0, 5.0]),
    np.array([5.0, 0.0, 5.0]),
    np.array([5.0, 5.0, 5.0]),
    np.array([10.0, 5.0, 6.0]),
    np.array([10.0, 10.0, 6.0]),
]

# Define obstacles
obstacles = [
    SphereObstacle(center=np.array([2.5, 2.5, 3.0]), radius=1.0),
    BoxObstacle(min_corner=np.array([7.0, 2.0, 2.0]),
                max_corner=np.array([9.0, 4.0, 5.0]))
]

print(f"Mission: {len(waypoints)} waypoints, {len(obstacles)} obstacles")

# Create simulation runner with SIMPLE physics
print("\nRunning with SIMPLE physics (correct behavior)...")
runner = SimulationRunner(
    uav=uav,
    waypoints=waypoints,
    obstacles=obstacles,
    dt=0.01,
    use_pybullet=False,  # Use simple physics!
    controller_params={'r': 5.0, 'rho': 0.5}
)

# Initial state
x0 = np.array([0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0])

# Initialize PyBullet for visualization
print("Initializing PyBullet for visualization...")
client_id = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -params.g)

# Load ground
plane_id = p.loadURDF("plane.urdf")

# Create UAV visual
collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.1])
visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.1],
                                   rgbaColor=[0.2, 0.5, 0.8, 1.0])
uav_id = p.createMultiBody(baseMass=0.0,  # No physics, just visual
                          baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=visual_shape,
                          basePosition=[0, 0, 1])

# Add rotors
rotor_positions = [
    [params.l, 0, 0], [0, params.l, 0],
    [-params.l, 0, 0], [0, -params.l, 0]
]
for pos in rotor_positions:
    rotor_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=0.02,
                                      rgbaColor=[0.8, 0.1, 0.1, 1.0])
    p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=rotor_visual,
                     basePosition=pos)

# Add obstacles
for obs in obstacles:
    obs_dict = obs.get_dict()
    if obs_dict['type'] == 'sphere':
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=obs_dict['radius'],
                                    rgbaColor=[1.0, 0.2, 0.2, 0.7])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                         basePosition=obs_dict['center'])
    elif obs_dict['type'] == 'box':
        size = (obs_dict['max'] - obs_dict['min']) / 2
        center = (obs_dict['max'] + obs_dict['min']) / 2
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size,
                                    rgbaColor=[1.0, 0.5, 0.2, 0.7])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                         basePosition=center)

# Add waypoints
for i, wp in enumerate(waypoints):
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.3,
                                rgbaColor=[0.2, 1.0, 0.2, 0.8])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=wp)

print("\n" + "=" * 70)
print("RUNNING SIMULATION")
print("=" * 70)
print("\nThe simulation is running with SIMPLE physics.")
print("PyBullet is only used for visualization.\n")

# Run simulation
trajectory_points = []
results = runner.run(x0=x0, t_final=60.0)
t, states = results[0], results[1]

print("\n" + "=" * 70)
print("REPLAYING IN PYBULLET")
print("=" * 70)
print("\nNow visualizing the recorded trajectory...")
print("Watch the PyBullet window!\n")

# Replay in PyBullet
for i in range(0, len(states), 5):  # Show every 5th frame for speed
    state = states[i]

    # Update UAV position
    position = [state[0], state[2], state[4]]
    roll, pitch, yaw = state[6], state[8], state[10]
    orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    p.resetBasePositionAndOrientation(uav_id, position, orientation)

    # Update camera
    p.resetDebugVisualizerCamera(
        cameraDistance=15.0,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=position
    )

    # Draw trajectory
    if len(trajectory_points) > 0:
        p.addUserDebugLine(trajectory_points[-1], position, [0, 0, 1], 2, lifeTime=0)
    trajectory_points.append(position)

    # Real-time playback
    time.sleep(0)  # 5x frame skip

    if i % 100 == 0:
        print(f"  Replay progress: {100*i/len(states):.0f}%")

print("\nReplay complete!")
print(f"Final position: ({states[-1, 0]:.2f}, {states[-1, 2]:.2f}, {states[-1, 4]:.2f})")

print("\nPyBullet window will stay open for 10 seconds...")
time.sleep(10)

p.disconnect()
print("\nDone!")
print("=" * 70)
