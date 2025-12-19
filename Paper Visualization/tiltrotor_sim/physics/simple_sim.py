"""
Simple numerical integrator (original implementation).
"""

import numpy as np
from ..models.dynamics import TiltRotorUAV


class SimpleSimulator:
    """Simple Euler integration simulator.

    This is the original lightweight simulator that uses Euler integration
    without physics engine. Useful for fast prototyping and testing.

    Attributes:
        uav (TiltRotorUAV): UAV dynamics model
        dt (float): Timestep
        state (np.ndarray): Current state
    """

    def __init__(self, uav: TiltRotorUAV, dt: float = 0.01):
        """Initialize simple simulator.

        Args:
            uav: UAV dynamics model
            dt: Integration timestep [s]
        """
        self.uav = uav
        self.dt = dt
        self.state = np.zeros(12)

    def reset(self, initial_state: np.ndarray):
        """Reset to initial state.

        Args:
            initial_state: Initial state [12]
        """
        self.state = initial_state.copy()

    def step(self, control: np.ndarray) -> np.ndarray:
        """Step simulation with control.

        Args:
            control: Control input [6 or 7]

        Returns:
            np.ndarray: New state [12]
        """
        self.state = self.uav.step(self.state, control, self.dt)
        return self.state

    def get_state(self) -> np.ndarray:
        """Get current state.

        Returns:
            np.ndarray: Current state [12]
        """
        return self.state.copy()
