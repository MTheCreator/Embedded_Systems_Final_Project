"""
Waypoint guidance controller with obstacle avoidance.
"""

import numpy as np
from enum import Enum
from typing import List, Tuple


class NavigationState(Enum):
    """Navigation state machine states."""
    NAVIGATE_TO_WAYPOINT = 1
    AVOID_OBSTACLE = 2
    HOVER_AT_WAYPOINT = 3
    MISSION_COMPLETE = 4


class WaypointGuidanceController:
    """Smooth altitude transitions with intermediate waypoints and obstacle avoidance.

    This controller implements a state machine for navigating through waypoints
    while avoiding obstacles. It manages altitude smoothly and provides
    desired position and velocity references.

    Attributes:
        waypoints (List[np.ndarray]): List of 3D waypoints
        approach_speed (float): Maximum horizontal approach speed [m/s]
        current_wp_idx (int): Index of current target waypoint
        state (NavigationState): Current navigation state
        hover_timer (float): Time spent hovering at current waypoint
        hover_duration (float): Required hover time before advancing
        current_altitude_target (float): Smoothed altitude target
    """

    def __init__(self, waypoints: List[np.ndarray], approach_speed: float = 1.5):
        """Initialize waypoint guidance controller.

        Args:
            waypoints: List of 3D waypoint positions
            approach_speed: Maximum horizontal velocity [m/s]
        """
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.approach_speed = approach_speed
        self.current_wp_idx = 0
        self.state = NavigationState.NAVIGATE_TO_WAYPOINT
        self.hover_timer = 0.0
        self.hover_duration = 0.5  # seconds
        self.current_altitude_target = 7.0  # Start at safe altitude

    def compute_desired_state(
        self, current_pos: np.ndarray, dt: float, obstacles: List
    ) -> Tuple[np.ndarray, np.ndarray, NavigationState]:
        """Compute desired position and velocity for navigation.

        Args:
            current_pos: Current UAV position [x, y, z]
            dt: Timestep [s]
            obstacles: List of obstacle objects or dictionaries

        Returns:
            tuple: (desired_position, desired_velocity, navigation_state)
        """
        # Check mission completion
        if self.current_wp_idx >= len(self.waypoints):
            self.state = NavigationState.MISSION_COMPLETE
            return current_pos, np.zeros(3), self.state

        target_wp = self.waypoints[self.current_wp_idx]
        horizontal_dist = np.linalg.norm(target_wp[:2] - current_pos[:2])
        total_dist = np.linalg.norm(target_wp - current_pos)

        # Waypoint achievement check
        if total_dist < 0.7:
            self.state = NavigationState.HOVER_AT_WAYPOINT
            self.hover_timer += dt

            if self.hover_timer >= self.hover_duration:
                print(f"  ✓✓✓ WAYPOINT {self.current_wp_idx + 1} REACHED!")
                self.current_wp_idx += 1
                self.hover_timer = 0.0

                if self.current_wp_idx >= len(self.waypoints):
                    self.state = NavigationState.MISSION_COMPLETE
                else:
                    self.state = NavigationState.NAVIGATE_TO_WAYPOINT

            return target_wp, np.zeros(3), self.state

        # Obstacle assessment
        min_obs_dist = self._get_min_obstacle_distance(current_pos, obstacles)
        is_path_clear = self._is_path_clear(current_pos, target_wp, obstacles)

        # Altitude management with smooth transitions
        max_altitude_change_rate = 0.3  # m/s

        if not is_path_clear or min_obs_dist < 2.0:
            # Climb to avoid obstacles
            self.state = NavigationState.AVOID_OBSTACLE
            safe_altitude = self._compute_safe_altitude(current_pos, target_wp, obstacles)
            target_altitude = safe_altitude
        else:
            # Path is clear
            self.state = NavigationState.NAVIGATE_TO_WAYPOINT

            # Gradually transition to waypoint altitude when horizontally close
            if horizontal_dist < 4.0:
                target_altitude = target_wp[2]
            else:
                # Maintain current altitude while traveling horizontally
                target_altitude = self.current_altitude_target

        # Smooth altitude transition (rate limiting)
        altitude_error = target_altitude - self.current_altitude_target
        if abs(altitude_error) > max_altitude_change_rate * dt:
            self.current_altitude_target += np.sign(altitude_error) * max_altitude_change_rate * dt
        else:
            self.current_altitude_target = target_altitude

        # Build desired position with smoothed altitude
        desired_position = np.array([
            target_wp[0],
            target_wp[1],
            self.current_altitude_target
        ])

        # Compute velocity toward waypoint
        direction = target_wp[:2] - current_pos[:2]
        h_dist = np.linalg.norm(direction)

        if h_dist > 1e-6:
            h_direction = direction / h_dist
            h_speed = min(self.approach_speed, h_dist / 2.0)
            h_speed = max(h_speed, 0.3)
        else:
            h_direction = np.zeros(2)
            h_speed = 0.0

        # Vertical velocity based on altitude target
        z_error = self.current_altitude_target - current_pos[2]
        z_velocity = 0.5 * z_error  # Proportional control
        z_velocity = np.clip(z_velocity, -0.8, 0.8)

        desired_velocity = np.array([
            h_speed * h_direction[0],
            h_speed * h_direction[1],
            z_velocity
        ])

        # Add gentle repulsion if very close to obstacles
        if min_obs_dist < 1.5:
            repulsion = self._compute_repulsion(current_pos, obstacles)
            desired_velocity[:2] += 0.3 * repulsion[:2]  # Only horizontal

        return desired_position, desired_velocity, self.state

    def _compute_safe_altitude(
        self, current_pos: np.ndarray, target_wp: np.ndarray, obstacles: List
    ) -> float:
        """Find safe altitude above obstacles.

        Args:
            current_pos: Current position
            target_wp: Target waypoint
            obstacles: List of obstacles

        Returns:
            float: Safe altitude [m]
        """
        safe_alt = 7.0

        for obs in obstacles:
            obs_dict = obs.get_dict() if hasattr(obs, 'get_dict') else obs

            if obs_dict['type'] == 'sphere':
                obs_top = obs_dict['center'][2] + obs_dict['radius']
                if self._is_obstacle_near_path(current_pos, target_wp, obs_dict):
                    safe_alt = max(safe_alt, obs_top + 2.5)
            elif obs_dict['type'] == 'box':
                obs_top = obs_dict['max'][2]
                if self._is_obstacle_near_path(current_pos, target_wp, obs_dict):
                    safe_alt = max(safe_alt, obs_top + 2.5)

        return safe_alt

    def _is_obstacle_near_path(
        self, start: np.ndarray, end: np.ndarray, obs: dict
    ) -> bool:
        """Check if obstacle is near the path.

        Args:
            start: Start position
            end: End position
            obs: Obstacle dictionary

        Returns:
            bool: True if obstacle is near path
        """
        if obs['type'] == 'sphere':
            obs_pos = obs['center'][:2]
        else:
            obs_pos = (obs['min'][:2] + obs['max'][:2]) / 2

        start_2d = start[:2]
        end_2d = end[:2]

        path_vec = end_2d - start_2d
        path_len = np.linalg.norm(path_vec)

        if path_len < 1e-6:
            return False

        to_obs = obs_pos - start_2d
        projection = np.dot(to_obs, path_vec) / path_len

        if 0 <= projection <= path_len:
            path_norm = path_vec / path_len
            perp = to_obs - projection * path_norm
            return np.linalg.norm(perp) < 3.0

        return False

    def _is_path_clear(
        self, start: np.ndarray, end: np.ndarray, obstacles: List
    ) -> bool:
        """Check if straight path is clear of obstacles.

        Args:
            start: Start position
            end: End position
            obstacles: List of obstacles

        Returns:
            bool: True if path is clear
        """
        n_samples = 20
        for i in range(n_samples):
            t = i / (n_samples - 1)
            point = start + t * (end - start)

            for obs in obstacles:
                obs_dict = obs.get_dict() if hasattr(obs, 'get_dict') else obs

                if obs_dict['type'] == 'sphere':
                    if np.linalg.norm(point - obs_dict['center']) < obs_dict['radius'] + 1.0:
                        return False
                elif obs_dict['type'] == 'box':
                    if (obs_dict['min'][0] - 1.0 <= point[0] <= obs_dict['max'][0] + 1.0 and
                        obs_dict['min'][1] - 1.0 <= point[1] <= obs_dict['max'][1] + 1.0 and
                        obs_dict['min'][2] - 1.0 <= point[2] <= obs_dict['max'][2] + 1.0):
                        return False
        return True

    def _get_min_obstacle_distance(
        self, position: np.ndarray, obstacles: List
    ) -> float:
        """Get minimum distance to any obstacle.

        Args:
            position: Current position
            obstacles: List of obstacles

        Returns:
            float: Minimum distance to obstacles
        """
        min_dist = float('inf')

        for obs in obstacles:
            if hasattr(obs, 'distance_to'):
                dist = obs.distance_to(position)
            else:
                # Dictionary format
                if obs['type'] == 'sphere':
                    dist = np.linalg.norm(position - obs['center']) - obs['radius']
                elif obs['type'] == 'box':
                    closest = np.clip(position, obs['min'], obs['max'])
                    dist = np.linalg.norm(position - closest)

            min_dist = min(min_dist, dist)

        return min_dist

    def _compute_repulsion(
        self, position: np.ndarray, obstacles: List
    ) -> np.ndarray:
        """Compute emergency repulsion when very close to obstacles.

        Args:
            position: Current position
            obstacles: List of obstacles

        Returns:
            np.ndarray: Repulsion vector
        """
        total = np.zeros(3)

        for obs in obstacles:
            obs_dict = obs.get_dict() if hasattr(obs, 'get_dict') else obs

            if obs_dict['type'] == 'sphere':
                vec = position - obs_dict['center']
                dist = np.linalg.norm(vec)
                clearance = dist - obs_dict['radius']

                if clearance < 1.5 and dist > 1e-6:
                    total += (2.0 / max(clearance, 0.2)) * (vec / dist)

            elif obs_dict['type'] == 'box':
                closest = np.clip(position, obs_dict['min'], obs_dict['max'])
                vec = position - closest
                dist = np.linalg.norm(vec)

                if dist < 1.5:
                    if dist > 1e-6:
                        total += (2.0 / max(dist, 0.2)) * (vec / dist)
                    else:
                        total += np.array([0, 0, 3.0])

        mag = np.linalg.norm(total)
        if mag > 3.0:
            total = 3.0 * total / mag

        return total

    def reset(self):
        """Reset guidance controller to initial state."""
        self.current_wp_idx = 0
        self.state = NavigationState.NAVIGATE_TO_WAYPOINT
        self.hover_timer = 0.0
        self.current_altitude_target = 7.0
