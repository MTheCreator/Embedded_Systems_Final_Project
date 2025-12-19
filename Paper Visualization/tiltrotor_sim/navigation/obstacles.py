"""
Obstacle representation for collision detection and avoidance.
"""

import numpy as np
from abc import ABC, abstractmethod


class Obstacle(ABC):
    """Abstract base class for obstacles."""

    @abstractmethod
    def distance_to(self, point: np.ndarray) -> float:
        """Calculate distance from point to obstacle surface.

        Args:
            point: 3D position [x, y, z]

        Returns:
            float: Distance to obstacle surface (negative if inside)
        """
        pass

    @abstractmethod
    def is_collision(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if point collides with obstacle.

        Args:
            point: 3D position
            margin: Safety margin

        Returns:
            bool: True if collision detected
        """
        pass

    @abstractmethod
    def get_dict(self) -> dict:
        """Get obstacle as dictionary for compatibility.

        Returns:
            dict: Obstacle parameters
        """
        pass


class SphereObstacle(Obstacle):
    """Spherical obstacle.

    Attributes:
        center (np.ndarray): Center position [x, y, z]
        radius (float): Sphere radius
    """

    def __init__(self, center: np.ndarray, radius: float):
        """Initialize sphere obstacle.

        Args:
            center: Center position [x, y, z]
            radius: Sphere radius
        """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)

    def distance_to(self, point: np.ndarray) -> float:
        """Calculate distance from point to sphere surface.

        Args:
            point: 3D position

        Returns:
            float: Distance to surface (negative if inside)
        """
        return np.linalg.norm(point - self.center) - self.radius

    def is_collision(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if point collides with sphere.

        Args:
            point: 3D position
            margin: Safety margin

        Returns:
            bool: True if collision
        """
        return self.distance_to(point) < margin

    def get_dict(self) -> dict:
        """Get sphere as dictionary.

        Returns:
            dict: {'type': 'sphere', 'center': [...], 'radius': ...}
        """
        return {
            'type': 'sphere',
            'center': self.center,
            'radius': self.radius
        }

    def __repr__(self):
        return f"SphereObstacle(center={self.center}, radius={self.radius})"


class BoxObstacle(Obstacle):
    """Box-shaped (rectangular) obstacle.

    Attributes:
        min_corner (np.ndarray): Minimum corner [x, y, z]
        max_corner (np.ndarray): Maximum corner [x, y, z]
    """

    def __init__(self, min_corner: np.ndarray, max_corner: np.ndarray):
        """Initialize box obstacle.

        Args:
            min_corner: Minimum corner position [x, y, z]
            max_corner: Maximum corner position [x, y, z]
        """
        self.min_corner = np.array(min_corner, dtype=float)
        self.max_corner = np.array(max_corner, dtype=float)

    def distance_to(self, point: np.ndarray) -> float:
        """Calculate distance from point to box surface.

        Args:
            point: 3D position

        Returns:
            float: Distance to surface (negative if inside)
        """
        # Find closest point on box
        closest = np.clip(point, self.min_corner, self.max_corner)

        # Check if point is inside
        if np.all(point >= self.min_corner) and np.all(point <= self.max_corner):
            # Inside: return negative distance to nearest face
            dist_to_faces = np.minimum(
                point - self.min_corner,
                self.max_corner - point
            )
            return -np.min(dist_to_faces)
        else:
            # Outside: return positive distance
            return np.linalg.norm(point - closest)

    def is_collision(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if point collides with box.

        Args:
            point: 3D position
            margin: Safety margin

        Returns:
            bool: True if collision
        """
        expanded_min = self.min_corner - margin
        expanded_max = self.max_corner + margin

        return np.all(point >= expanded_min) and np.all(point <= expanded_max)

    def get_center(self) -> np.ndarray:
        """Get box center.

        Returns:
            np.ndarray: Center position
        """
        return (self.min_corner + self.max_corner) / 2

    def get_size(self) -> np.ndarray:
        """Get box dimensions.

        Returns:
            np.ndarray: Size [dx, dy, dz]
        """
        return self.max_corner - self.min_corner

    def get_dict(self) -> dict:
        """Get box as dictionary.

        Returns:
            dict: {'type': 'box', 'min': [...], 'max': [...]}
        """
        return {
            'type': 'box',
            'min': self.min_corner,
            'max': self.max_corner
        }

    def __repr__(self):
        return f"BoxObstacle(min={self.min_corner}, max={self.max_corner})"
