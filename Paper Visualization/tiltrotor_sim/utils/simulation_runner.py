"""
High-level simulation runner orchestrating all components.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..models.dynamics import TiltRotorUAV
from ..models.linearization import SystemLinearization
from ..control.hinf_controller import HInfinityController
from ..control.kalman_filter import HInfinityKalmanFilter
from ..navigation.guidance import WaypointGuidanceController
from ..physics.simple_sim import SimpleSimulator
from ..physics.pybullet_sim import PyBulletSimulator


class SimulationRunner:
    """High-level simulation orchestrator.

    This class integrates all components and runs the complete simulation loop:
    - Physics simulation (PyBullet or simple)
    - Control system (H-infinity + Kalman filter)
    - Navigation and guidance
    - Data logging

    Attributes:
        uav (TiltRotorUAV): UAV dynamics model
        linearizer (SystemLinearization): Linearization module
        controller (HInfinityController): H-infinity controller
        kalman_filter (HInfinityKalmanFilter): State estimator
        guidance (WaypointGuidanceController): Waypoint guidance
        simulator: Physics simulator (PyBullet or simple)
        use_pybullet (bool): Whether using PyBullet
    """

    def __init__(self, uav: TiltRotorUAV, waypoints: List[np.ndarray],
                 obstacles: List = None, dt: float = 0.01,
                 use_pybullet: bool = False, gui: bool = True,
                 controller_params: dict = None):
        """Initialize simulation runner.

        Args:
            uav: UAV dynamics model
            waypoints: List of waypoint positions
            obstacles: List of obstacles (optional)
            dt: Simulation timestep [s]
            use_pybullet: Use PyBullet physics engine
            gui: Show PyBullet GUI (if use_pybullet=True)
            controller_params: Controller parameters dict (r, rho)
        """
        self.uav = uav
        self.linearizer = SystemLinearization(uav)
        self.dt = dt
        self.use_pybullet = use_pybullet

        # Controller
        if controller_params is None:
            controller_params = {'r': 5.0, 'rho': 0.5}

        self.controller = HInfinityController(**controller_params)

        # Kalman filter
        self.kalman_filter = HInfinityKalmanFilter()

        # Guidance
        self.guidance = WaypointGuidanceController(waypoints, approach_speed=1.5)

        # Obstacles
        self.obstacles = obstacles if obstacles is not None else []

        # Simulator
        if use_pybullet:
            self.simulator = PyBulletSimulator(uav.params, dt=dt, gui=gui)
            # Add obstacles to PyBullet
            for obs in self.obstacles:
                self.simulator.add_obstacle(obs)
            # Add waypoint markers
            for i, wp in enumerate(waypoints):
                self.simulator.add_waypoint_marker(wp, i)
        else:
            self.simulator = SimpleSimulator(uav, dt=dt)

    def run(self, x0: np.ndarray, t_final: float = 100.0,
            Q_matrix: np.ndarray = None) -> Tuple:
        """Run complete simulation.

        Args:
            x0: Initial state [12]
            t_final: Simulation duration [s]
            Q_matrix: State weighting matrix for controller (optional)

        Returns:
            tuple: (t, states, states_est, controls, references, wp_indices,
                   collision_flags, state_history)
        """
        n_steps = int(t_final / self.dt)

        # Default Q matrix with strong z tracking
        if Q_matrix is None:
            Q_matrix = np.diag([100, 150,   # X position and velocity
                               100, 150,   # Y position and velocity
                               200, 300,   # Z position and velocity (strong!)
                               30, 5,      # Roll and roll rate
                               30, 5,      # Pitch and pitch rate
                               20, 2])     # Yaw and yaw rate

        # Storage arrays
        t = np.zeros(n_steps)
        states = np.zeros((n_steps, 12))
        states_est = np.zeros((n_steps, 12))
        controls = np.zeros((n_steps, 6))
        references = np.zeros((n_steps, 12))
        wp_indices = np.zeros(n_steps)
        collision_flags = np.zeros(n_steps)
        state_history = []

        # Initialize
        self.simulator.reset(x0)
        self.kalman_filter.reset(x0)
        x = x0.copy()
        u = np.zeros(6)

        print("=" * 70)
        print("TILTROTOR UAV SIMULATION")
        print("=" * 70)
        print(f"Simulator: {'PyBullet' if self.use_pybullet else 'Simple'}")
        print(f"Timestep: {self.dt}s")
        print(f"Duration: {t_final}s")
        print(f"Waypoints: {len(self.guidance.waypoints)}")
        print(f"Obstacles: {len(self.obstacles)}")
        print()

        for i in range(n_steps):
            t[i] = i * self.dt
            current_pos = np.array([x[0], x[2], x[4]])

            # Guidance
            desired_pos, desired_vel, nav_state = self.guidance.compute_desired_state(
                current_pos, self.dt, self.obstacles
            )
            state_history.append(nav_state.value)

            # Build reference state
            x_ref = np.zeros(12)
            x_ref[0] = desired_pos[0]
            x_ref[1] = desired_vel[0]
            x_ref[2] = desired_pos[1]
            x_ref[3] = desired_vel[1]
            x_ref[4] = desired_pos[2]
            x_ref[5] = desired_vel[2]

            # Status print every 0.5 seconds
            if i % 50 == 0:
                if self.guidance.current_wp_idx < len(self.guidance.waypoints):
                    print(f"  t={t[i]:.1f}s: {nav_state.name}, "
                          f"WP{self.guidance.current_wp_idx + 1}, "
                          f"z={x[4]:.2f}, z_des={x_ref[4]:.2f}")

            # Check collision
            collision = False
            if self.use_pybullet:
                collision = self.simulator.check_collision()
            else:
                for obs in self.obstacles:
                    if hasattr(obs, 'is_collision'):
                        if obs.is_collision(current_pos, margin=0.3):
                            collision = True
                            break

            collision_flags[i] = 1.0 if collision else 0.0

            if collision:
                print(f"  ⚠️ COLLISION at t={t[i]:.1f}s!")

            # Store data
            states[i, :] = x
            wp_indices[i] = self.guidance.current_wp_idx
            references[i, :] = x_ref

            # Linearize and compute control
            A, B = self.linearizer.linearize(x, u[:6] if len(u) == 7 else u)
            self.controller.solve_riccati(A, B, Q_matrix)
            self.controller.compute_gain(B)

            error = x - x_ref
            altitude_error = x[4] - x_ref[4]
            u = self.controller.control_law(error, altitude_error)

            # Store only first 6 controls for compatibility
            controls[i, :] = u[:6]

            # Step simulation
            x = self.simulator.step(u)

            # State estimation
            y_true = self.kalman_filter.C @ x
            y_meas = y_true + np.random.multivariate_normal(
                np.zeros(self.kalman_filter.m), self.kalman_filter.R
            )

            self.kalman_filter.predict(A, B, u[:6], self.dt)
            self.kalman_filter.update(y_meas)
            states_est[i, :] = self.kalman_filter.get_estimate()

            # Check for mission completion
            if nav_state.value == 4:  # MISSION_COMPLETE
                print(f"\n  ★★★ MISSION COMPLETE at t={t[i]:.1f}s! ★★★\n")
                # Trim arrays
                actual_steps = i + 1
                t = t[:actual_steps]
                states = states[:actual_steps]
                states_est = states_est[:actual_steps]
                controls = controls[:actual_steps]
                references = references[:actual_steps]
                wp_indices = wp_indices[:actual_steps]
                collision_flags = collision_flags[:actual_steps]
                state_history = state_history[:actual_steps]
                break

        return (t, states, states_est, controls, references,
                wp_indices, collision_flags, state_history)

    def close(self):
        """Clean up simulator resources."""
        if self.use_pybullet and hasattr(self.simulator, 'close'):
            self.simulator.close()
