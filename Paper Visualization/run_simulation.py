#!/usr/bin/env python3
"""
Standalone script to run tiltrotor UAV simulation.

This script demonstrates how to use the modular simulation framework
without a notebook environment.
"""

import numpy as np
import argparse
from tiltrotor_sim.models import TiltRotorUAVParameters, TiltRotorUAV
from tiltrotor_sim.navigation import SphereObstacle, BoxObstacle
from tiltrotor_sim.utils import SimulationRunner
from tiltrotor_sim.visualization import SimulationPlotter


def main():
    parser = argparse.ArgumentParser(description='Run tiltrotor UAV simulation')
    parser.add_argument('--pybullet', action='store_true',
                       help='Use PyBullet physics engine (requires PyBullet)')
    parser.add_argument('--gui', action='store_true',
                       help='Show PyBullet GUI (only with --pybullet)')
    parser.add_argument('--time', type=float, default=100.0,
                       help='Simulation duration [s]')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Simulation timestep [s]')
    parser.add_argument('--save-prefix', type=str, default='results',
                       help='Prefix for saved plots')

    args = parser.parse_args()

    print("=" * 70)
    print("TILTROTOR UAV SIMULATION")
    print("=" * 70)

    # Create UAV
    print("\n1. Creating UAV model...")
    params = TiltRotorUAVParameters()
    uav = TiltRotorUAV(params)
    print(f"   ✓ UAV: m={params.m}kg, hover_thrust={params.get_total_hover_thrust():.2f}N")

    # Define mission
    print("\n2. Defining mission...")
    waypoints = [
        np.array([2.0, 2.0, 7.0]),
        np.array([7.0, 7.0, 7.0]),
        np.array([15.0, 8.0, 6.0]),
        np.array([20.0, 10.0, 6.0])
    ]

    obstacles = [
        SphereObstacle(center=np.array([5.0, 5.0, 3.0]), radius=1.5),
        BoxObstacle(min_corner=np.array([2.0, 2.0, 2.0]),
                   max_corner=np.array([9.0, 4.0, 5.0]))
    ]

    print(f"   ✓ {len(waypoints)} waypoints, {len(obstacles)} obstacles")

    # Create simulation runner
    print("\n3. Initializing simulator...")
    runner = SimulationRunner(
        uav=uav,
        waypoints=waypoints,
        obstacles=obstacles,
        dt=args.dt,
        use_pybullet=args.pybullet,
        gui=args.gui if args.pybullet else False,
        controller_params={'r': 5.0, 'rho': 0.5}
    )
    print(f"   ✓ Simulator: {'PyBullet' if args.pybullet else 'Simple Euler'}")

    # Initial state
    x0 = np.array([0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Run simulation
    print(f"\n4. Running simulation ({args.time}s)...")
    results = runner.run(x0=x0, t_final=args.time)
    t, states, states_est, controls, references, wp_indices, collision_flags, state_history = results

    # Clean up
    if args.pybullet:
        runner.close()

    print("\n5. Simulation complete!")
    print(f"   ✓ Mission time: {t[-1]:.2f}s")
    print(f"   ✓ Final position: ({states[-1, 0]:.2f}, {states[-1, 2]:.2f}, {states[-1, 4]:.2f})")
    print(f"   ✓ Collisions: {int(np.sum(collision_flags))}")

    # Performance metrics
    position_error = np.linalg.norm(states[:, [0, 2, 4]] - references[:, [0, 2, 4]], axis=1)
    print(f"\n6. Performance Metrics:")
    print(f"   Mean position error: {np.mean(position_error):.3f} m")
    print(f"   Max position error: {np.max(position_error):.3f} m")
    print(f"   Max velocity: {np.max(np.linalg.norm(states[:, [1, 3, 5]], axis=1)):.3f} m/s")

    # Visualization
    print("\n7. Generating plots...")

    print("   - State trajectories...")
    SimulationPlotter.plot_states(
        t, states, states_est, references,
        save_path=f'{args.save_prefix}_states.png'
    )

    print("   - Control inputs...")
    SimulationPlotter.plot_controls(
        t, controls,
        save_path=f'{args.save_prefix}_controls.png'
    )

    print("   - 3D trajectory...")
    SimulationPlotter.plot_3d_trajectory(
        states, references, waypoints, obstacles,
        save_path=f'{args.save_prefix}_3d.png'
    )

    print("   - Mission summary...")
    SimulationPlotter.plot_mission_summary(
        t, states, references, wp_indices, collision_flags, state_history,
        save_path=f'{args.save_prefix}_summary.png'
    )

    print("\nAll done! Plots saved with prefix:", args.save_prefix)
    print("=" * 70)


if __name__ == '__main__':
    main()
