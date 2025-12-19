"""
Plotting utilities for simulation visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from typing import List, Optional


class SimulationPlotter:
    """Comprehensive plotting for simulation results.

    This class provides methods to visualize:
    - State trajectories
    - Control inputs
    - 3D flight paths
    - Mission progress
    - Tracking errors
    """

    @staticmethod
    def plot_states(t: np.ndarray, states: np.ndarray, states_est: np.ndarray = None,
                   references: np.ndarray = None, save_path: str = None):
        """Plot all state variables over time.

        Args:
            t: Time vector [n]
            states: State history [n×12]
            states_est: Estimated states [n×12] (optional)
            references: Reference states [n×12] (optional)
            save_path: Path to save figure (optional)
        """
        labels = ['x [m]', 'vx [m/s]', 'y [m]', 'vy [m/s]', 'z [m]', 'vz [m/s]',
                  'φ [rad]', 'ωφ [rad/s]', 'θ [rad]', 'ωθ [rad/s]', 'ψ [rad]', 'ωψ [rad/s]']

        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i in range(12):
            axes[i].plot(t, states[:, i], 'b-', label='True', linewidth=1.5)

            if states_est is not None:
                axes[i].plot(t, states_est[:, i], 'g--', label='Estimated', linewidth=1)

            if references is not None:
                axes[i].plot(t, references[:, i], 'r:', label='Reference', linewidth=1.5)

            axes[i].set_xlabel('Time [s]')
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_controls(t: np.ndarray, controls: np.ndarray, save_path: str = None):
        """Plot control inputs over time.

        Args:
            t: Time vector [n]
            controls: Control history [n×6]
            save_path: Path to save figure (optional)
        """
        control_labels = ['θ1 [rad]', 'θ2 [rad]', 'θ3 [rad]',
                         'θ4 [rad]', 'τ1 [Nm]', 'τ2 [Nm]']

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i in range(6):
            axes[i].plot(t, controls[:, i], 'b-', linewidth=1.5)
            axes[i].set_xlabel('Time [s]')
            axes[i].set_ylabel(control_labels[i])
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'Control Input {control_labels[i]}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_3d_trajectory(states: np.ndarray, references: np.ndarray = None,
                          waypoints: List[np.ndarray] = None,
                          obstacles: List = None,
                          save_path: str = None):
        """Plot 3D trajectory with orientation, waypoints, and obstacles.

        Args:
            states: State history [n×12]
            references: Reference trajectory [n×12] (optional)
            waypoints: List of waypoint positions (optional)
            obstacles: List of obstacle dictionaries (optional)
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(states[:, 0], states[:, 2], states[:, 4],
               'b-', linewidth=2, label='Actual', alpha=0.6)

        if references is not None:
            ax.plot(references[:, 0], references[:, 2], references[:, 4],
                   'r--', linewidth=2, label='Reference', alpha=0.4)

        # Plot start and end
        ax.scatter(states[0, 0], states[0, 2], states[0, 4],
                  c='g', s=100, marker='o', label='Start', edgecolors='black')
        ax.scatter(states[-1, 0], states[-1, 2], states[-1, 4],
                  c='r', s=100, marker='*', label='End', edgecolors='black')

        # Plot waypoints
        if waypoints is not None:
            waypoints_array = np.array(waypoints)
            ax.scatter(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2],
                      c='green', marker='*', s=200, label='Waypoints',
                      edgecolors='black', linewidth=2)

            for i, wp in enumerate(waypoints):
                ax.text(wp[0], wp[1], wp[2] + 0.5, f'WP{i+1}',
                       fontsize=10, color='green', weight='bold')

        # Plot obstacles
        if obstacles is not None:
            for obs in obstacles:
                obs_dict = obs.get_dict() if hasattr(obs, 'get_dict') else obs
                SimulationPlotter._plot_obstacle(ax, obs_dict)

        # Plot orientation axes at selected points
        SimulationPlotter._plot_orientations(ax, states)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.title('3D Trajectory with Orientation')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    @staticmethod
    def _plot_obstacle(ax, obs_dict: dict):
        """Plot obstacle on 3D axis.

        Args:
            ax: Matplotlib 3D axis
            obs_dict: Obstacle dictionary
        """
        if obs_dict['type'] == 'sphere':
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = obs_dict['center'][0] + obs_dict['radius'] * np.outer(np.cos(u), np.sin(v))
            y = obs_dict['center'][1] + obs_dict['radius'] * np.outer(np.sin(u), np.sin(v))
            z = obs_dict['center'][2] + obs_dict['radius'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='red', alpha=0.3)

        elif obs_dict['type'] == 'box':
            min_corner = obs_dict['min']
            max_corner = obs_dict['max']

            vertices = np.array([
                [min_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
                [min_corner[0], max_corner[1], max_corner[2]]
            ])

            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]]
            ]

            box = Poly3DCollection(faces, alpha=0.3, linewidths=1, edgecolor='r')
            box.set_facecolor('orange')
            ax.add_collection3d(box)

    @staticmethod
    def _rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
        """Compute rotation matrix from Euler angles (ZYX convention).

        Args:
            phi: Roll angle [rad]
            theta: Pitch angle [rad]
            psi: Yaw angle [rad]

        Returns:
            np.ndarray: 3×3 rotation matrix
        """
        # Roll around x-axis
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])

        # Pitch around y-axis
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])

        # Yaw around z-axis
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])

        return R_z @ R_y @ R_x

    @staticmethod
    def _plot_orientations(ax, states: np.ndarray, skip: int = None):
        """Plot orientation axes along trajectory.

        Args:
            ax: Matplotlib 3D axis
            states: State history [n×12]
            skip: Plot every skip-th frame (auto if None)
        """
        n_points = len(states)
        if skip is None:
            skip = max(1, n_points // 20)

        axis_length = 0.5

        for i in range(0, n_points, skip):
            x_pos = states[i, 0]
            y_pos = states[i, 2]
            z_pos = states[i, 4]

            phi = states[i, 6]    # Roll
            theta = states[i, 8]  # Pitch
            psi = states[i, 10]   # Yaw

            # Get rotation matrix
            R = SimulationPlotter._rotation_matrix(phi, theta, psi)

            # Body frame axes in inertial frame
            x_axis = R @ np.array([axis_length, 0, 0])
            y_axis = R @ np.array([0, axis_length, 0])
            z_axis = R @ np.array([0, 0, axis_length])

            # Plot axes
            ax.plot([x_pos, x_pos + x_axis[0]],
                   [y_pos, y_pos + x_axis[1]],
                   [z_pos, z_pos + x_axis[2]],
                   'r-', linewidth=1.5, alpha=0.7)

            ax.plot([x_pos, x_pos + y_axis[0]],
                   [y_pos, y_pos + y_axis[1]],
                   [z_pos, z_pos + y_axis[2]],
                   'g-', linewidth=1.5, alpha=0.7)

            ax.plot([x_pos, x_pos + z_axis[0]],
                   [y_pos, y_pos + z_axis[1]],
                   [z_pos, z_pos + z_axis[2]],
                   'c-', linewidth=1.5, alpha=0.7)

    @staticmethod
    def plot_mission_summary(t: np.ndarray, states: np.ndarray, references: np.ndarray,
                            wp_indices: np.ndarray, collision_flags: np.ndarray,
                            state_history: List = None, save_path: str = None):
        """Plot mission summary with multiple subplots.

        Args:
            t: Time vector
            states: State history
            references: Reference trajectory
            wp_indices: Waypoint index history
            collision_flags: Collision detection flags
            state_history: Navigation state history (optional)
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=(15, 10))

        # Waypoint progress
        ax1 = fig.add_subplot(221)
        ax1.plot(t, wp_indices, 'b-', linewidth=2)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Current Waypoint Index')
        ax1.set_title('Waypoint Progress')
        ax1.grid(True, alpha=0.3)

        # State machine states
        ax2 = fig.add_subplot(222)
        if state_history is not None:
            ax2.plot(t, state_history[:len(t)], 'g-', linewidth=2)
            ax2.set_ylabel('State (1=NAV, 2=AVOID, 3=HOVER, 4=COMPLETE)')
            ax2.set_title('Navigation State')
        else:
            ax2.plot(t, collision_flags, 'r-', linewidth=2)
            ax2.set_ylabel('Collision Flag')
            ax2.set_title('Collision Detection')
            ax2.set_ylim(-0.1, 1.1)

        ax2.set_xlabel('Time [s]')
        ax2.grid(True, alpha=0.3)

        # Position errors
        ax3 = fig.add_subplot(223)
        pos_errors = np.linalg.norm(states[:, [0, 2, 4]] - references[:, [0, 2, 4]], axis=1)
        ax3.plot(t, pos_errors, 'purple', linewidth=2)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Position Error [m]')
        ax3.set_title('Tracking Position Error')
        ax3.grid(True, alpha=0.3)

        # Velocity magnitude
        ax4 = fig.add_subplot(224)
        vel_mag = np.linalg.norm(states[:, [1, 3, 5]], axis=1)
        ax4.plot(t, vel_mag, 'orange', linewidth=2)
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Speed [m/s]')
        ax4.set_title('UAV Speed')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
