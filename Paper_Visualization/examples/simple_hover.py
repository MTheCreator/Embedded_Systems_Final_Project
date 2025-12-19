#!/usr/bin/env python3
"""
Example: Simple hover test.

This script demonstrates basic usage by making the UAV hover at a fixed position.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tiltrotor_sim.models import TiltRotorUAVParameters, TiltRotorUAV
from tiltrotor_sim.utils import SimulationRunner
from tiltrotor_sim.visualization import SimulationPlotter


def main():
    print("=" * 70)
    print("SIMPLE HOVER TEST")
    print("=" * 70)

    # Create UAV
    print("\nCreating UAV...")
    params = TiltRotorUAVParameters()
    uav = TiltRotorUAV(params)
    print(f"✓ UAV created: m={params.m}kg")

    # Single waypoint at hover position
    waypoints = [np.array([0.0, 0.0, 5.0])]
    obstacles = []

    print(f"\nMission: Hover at (0, 0, 5)")

    # Create runner
    runner = SimulationRunner(
        uav=uav,
        waypoints=waypoints,
        obstacles=obstacles,
        dt=0.01,
        use_pybullet=False,
        controller_params={'r': 5.0, 'rho': 0.5}
    )

    # Initial state: start at ground level
    x0 = np.array([0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0])

    print("\nRunning simulation for 20 seconds...")
    results = runner.run(x0=x0, t_final=20.0)
    t, states, states_est, controls, references = results[:5]

    print(f"\n✓ Simulation complete!")
    print(f"  Start position: ({states[0, 0]:.2f}, {states[0, 2]:.2f}, {states[0, 4]:.2f})")
    print(f"  Final position: ({states[-1, 0]:.2f}, {states[-1, 2]:.2f}, {states[-1, 4]:.2f})")
    print(f"  Target position: (0.00, 0.00, 5.00)")

    # Calculate settling time (when position error < 0.1m)
    position_error = np.linalg.norm(states[:, [0, 2, 4]] - references[:, [0, 2, 4]], axis=1)
    settled_idx = np.where(position_error < 0.1)[0]
    if len(settled_idx) > 0:
        settling_time = t[settled_idx[0]]
        print(f"  Settling time: {settling_time:.2f}s (error < 0.1m)")

    # Plot results
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Z position
    ax = axes[0, 0]
    ax.plot(t, states[:, 4], 'b-', linewidth=2, label='Actual')
    ax.plot(t, references[:, 4], 'r--', linewidth=2, label='Reference')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Z Position [m]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Altitude Tracking')

    # X-Y position
    ax = axes[0, 1]
    ax.plot(states[:, 0], states[:, 2], 'b-', linewidth=2, label='Actual')
    ax.plot(0, 0, 'r*', markersize=15, label='Target')
    ax.plot(states[0, 0], states[0, 2], 'go', markersize=10, label='Start')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Horizontal Position')
    ax.axis('equal')

    # Position error
    ax = axes[1, 0]
    ax.plot(t, position_error, 'purple', linewidth=2)
    ax.axhline(y=0.1, color='r', linestyle='--', label='Tolerance (0.1m)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Error [m]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Tracking Error')
    ax.set_yscale('log')

    # Attitude angles
    ax = axes[1, 1]
    ax.plot(t, np.rad2deg(states[:, 6]), 'r-', linewidth=1.5, label='Roll')
    ax.plot(t, np.rad2deg(states[:, 8]), 'g-', linewidth=1.5, label='Pitch')
    ax.plot(t, np.rad2deg(states[:, 10]), 'b-', linewidth=1.5, label='Yaw')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Attitude Angles')

    plt.tight_layout()
    plt.savefig('hover_test_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to: hover_test_results.png")
    plt.show()

    print("\n✅ Test complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
