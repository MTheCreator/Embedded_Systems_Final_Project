#!/usr/bin/env python3
"""
Example: Compare Simple vs PyBullet physics simulation.

This script runs the same mission with both physics engines and compares results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tiltrotor_sim.models import TiltRotorUAVParameters, TiltRotorUAV
from tiltrotor_sim.navigation import SphereObstacle
from tiltrotor_sim.utils import SimulationRunner


def main():
    print("=" * 70)
    print("PHYSICS ENGINE COMPARISON")
    print("=" * 70)

    # Setup
    params = TiltRotorUAVParameters()
    uav = TiltRotorUAV(params)

    waypoints = [
        np.array([0.0, 0.0, 5.0]),
        np.array([5.0, 5.0, 5.0]),
        np.array([10.0, 0.0, 5.0]),
    ]

    obstacles = [
        SphereObstacle(center=np.array([5.0, 2.5, 3.0]), radius=1.0)
    ]

    x0 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    t_final = 30.0

    # Run with Simple physics
    print("\n1. Running with Simple Euler integrator...")
    runner_simple = SimulationRunner(
        uav, waypoints, obstacles,
        dt=0.01,
        use_pybullet=False
    )

    results_simple = runner_simple.run(x0, t_final)
    t_simple, states_simple = results_simple[0], results_simple[1]
    print(f"   ✓ Completed in {t_simple[-1]:.2f}s")

    # Run with PyBullet
    print("\n2. Running with PyBullet physics...")
    try:
        runner_pybullet = SimulationRunner(
            uav, waypoints, obstacles,
            dt=0.01,
            use_pybullet=True,
            gui=False  # No GUI for comparison
        )

        results_pybullet = runner_pybullet.run(x0, t_final)
        t_pb, states_pb = results_pybullet[0], results_pybullet[1]
        runner_pybullet.close()
        print(f"   ✓ Completed in {t_pb[-1]:.2f}s")

        has_pybullet = True
    except ImportError:
        print("   ⚠️  PyBullet not installed, skipping comparison")
        has_pybullet = False

    # Compare results
    if has_pybullet:
        print("\n3. Comparing results...")

        # Interpolate PyBullet results to match Simple timesteps
        states_pb_interp = np.zeros_like(states_simple)
        for i in range(12):
            states_pb_interp[:, i] = np.interp(t_simple, t_pb, states_pb[:, i])

        # Compute differences
        diff = states_simple - states_pb_interp
        position_diff = np.linalg.norm(diff[:, [0, 2, 4]], axis=1)

        print(f"   Mean position difference: {np.mean(position_diff):.4f} m")
        print(f"   Max position difference: {np.max(position_diff):.4f} m")
        print(f"   RMS position difference: {np.sqrt(np.mean(position_diff**2)):.4f} m")

        # Plot comparison
        print("\n4. Generating comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Position comparison
        ax = axes[0, 0]
        ax.plot(t_simple, states_simple[:, 0], 'b-', label='Simple - X', linewidth=2)
        ax.plot(t_pb, states_pb[:, 0], 'r--', label='PyBullet - X', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('X Position [m]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('X Position Comparison')

        ax = axes[0, 1]
        ax.plot(t_simple, states_simple[:, 4], 'b-', label='Simple - Z', linewidth=2)
        ax.plot(t_pb, states_pb[:, 4], 'r--', label='PyBullet - Z', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Z Position [m]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Z Position Comparison')

        # 2D trajectory
        ax = axes[1, 0]
        ax.plot(states_simple[:, 0], states_simple[:, 2], 'b-', label='Simple', linewidth=2)
        ax.plot(states_pb[:, 0], states_pb[:, 2], 'r--', label='PyBullet', linewidth=2)
        for wp in waypoints:
            ax.plot(wp[0], wp[1], 'g*', markersize=15)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('2D Trajectory Comparison')
        ax.axis('equal')

        # Position difference
        ax = axes[1, 1]
        ax.plot(t_simple, position_diff, 'purple', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position Difference [m]')
        ax.grid(True, alpha=0.3)
        ax.set_title('Position Difference (Simple - PyBullet)')

        plt.tight_layout()
        plt.savefig('physics_comparison.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved to: physics_comparison.png")
        plt.show()

    else:
        # Just plot Simple results
        print("\n3. Plotting Simple physics results...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(t_simple, states_simple[:, 0], 'b-', label='X', linewidth=2)
        ax.plot(t_simple, states_simple[:, 2], 'r-', label='Y', linewidth=2)
        ax.plot(t_simple, states_simple[:, 4], 'g-', label='Z', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Position vs Time')

        ax = axes[1]
        ax.plot(states_simple[:, 0], states_simple[:, 2], 'b-', linewidth=2)
        for wp in waypoints:
            ax.plot(wp[0], wp[1], 'g*', markersize=15)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.3)
        ax.set_title('2D Trajectory')
        ax.axis('equal')

        plt.tight_layout()
        plt.savefig('simple_physics_results.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved to: simple_physics_results.png")
        plt.show()

    print("\n✅ Comparison complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
