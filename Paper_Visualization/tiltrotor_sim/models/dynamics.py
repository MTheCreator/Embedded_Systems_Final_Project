"""
Nonlinear dynamics model for tilt-rotor UAV.
"""

import numpy as np
from .parameters import TiltRotorUAVParameters


class TiltRotorUAV:
    """Nonlinear dynamic model of tilt-rotor UAV.

    This class implements the full nonlinear equations of motion for a
    quadrotor with independently tilting rotors. The state vector is:
    x = [x, ẋ, y, ẏ, z, ż, φ, φ̇, θ, θ̇, ψ, ψ̇]ᵀ

    The control vector is:
    u = [θ₁, θ₂, θ₃, θ₄, τ₁, τ₂, δ_thrust]ᵀ

    where θᵢ are rotor tilt angles, τ₁, τ₂ are auxiliary torques,
    and δ_thrust is optional thrust modulation.

    Attributes:
        params (TiltRotorUAVParameters): Physical parameters
    """

    def __init__(self, params: TiltRotorUAVParameters):
        """Initialize UAV dynamics model.

        Args:
            params: Physical parameters of the UAV
        """
        self.params = params

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state derivative: ẋ = f(x, u).

        Implements the nonlinear differential equations from the paper.
        Supports both 6-input (standard) and 7-input (with thrust modulation).

        Args:
            x: State vector [12] - [x, ẋ, y, ẏ, z, ż, φ, φ̇, θ, θ̇, ψ, ψ̇]
            u: Control vector [6 or 7] - [θ₁, θ₂, θ₃, θ₄, τ₁, τ₂] or
               [θ₁, θ₂, θ₃, θ₄, τ₁, τ₂, δ_thrust]

        Returns:
            np.ndarray: State derivative ẋ [12]
        """
        p = self.params

        # Extract and clip controls
        if len(u) == 6:
            theta1, theta2, theta3, theta4, tau1, tau2 = u
            delta_thrust = 0.0
        else:
            theta1, theta2, theta3, theta4, tau1, tau2, delta_thrust = u

        # Clip tilt angles to ±17° (≈0.3 rad)
        theta1 = np.clip(theta1, -0.3, 0.3)
        theta2 = np.clip(theta2, -0.3, 0.3)
        theta3 = np.clip(theta3, -0.3, 0.3)
        theta4 = np.clip(theta4, -0.3, 0.3)

        # Clip auxiliary torques
        tau1 = np.clip(tau1, -2.0, 2.0)
        tau2 = np.clip(tau2, -2.0, 2.0)

        # Clip thrust modulation
        delta_thrust = np.clip(delta_thrust, -0.3, 0.3)

        # Extract states
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = x

        # Clip attitude angles for stability
        x11 = np.clip(x11, -np.pi, np.pi)  # Yaw
        x7 = np.clip(x7, -np.pi / 4, np.pi / 4)  # Roll
        x9 = np.clip(x9, -np.pi / 4, np.pi / 4)  # Pitch

        # Trigonometric functions
        c11 = np.cos(x11)
        s11 = np.sin(x11)

        # Compute modulated thrust forces
        thrust_scale = 1.0 + delta_thrust
        F1 = p.F1 * thrust_scale
        F2 = p.F2 * thrust_scale
        F3 = p.F3 * thrust_scale
        F4 = p.F4 * thrust_scale
        F_sum = F1 + F2 + F3 + F4

        # Position derivatives (velocity states)
        dx1 = x2
        dx3 = x4
        dx5 = x6

        # Angular position derivatives (angular velocity states)
        dx7 = x8
        dx9 = x10
        dx11 = x12

        # Linear acceleration in x (Eq. 28 from paper)
        dx2 = (1 / p.m) * (
            F1 * theta1 * c11 - F3 * theta3 * c11 -
            F4 * theta4 * c11 * x7 * x9 + F4 * theta4 * s11 +
            F2 * theta2 * c11 * x7 * x9 - F2 * theta2 * s11 +
            F_sum * c11 * x9 + F_sum * s11 * x7 - p.C1 * x2
        )

        # Linear acceleration in y (Eq. 30 from paper)
        dx4 = (1 / p.m) * (
            F1 * theta1 * s11 - F3 * theta3 * s11 -
            F4 * theta4 * s11 * x7 * x9 + F2 * theta2 * s11 * x7 * x9 -
            F4 * theta4 * c11 + F2 * theta2 * c11 +
            F_sum * s11 * x9 - F_sum * c11 * x7 - p.C2 * x4
        )

        # Linear acceleration in z (Eq. 32 from paper)
        dx6 = (1 / p.m) * (
            -F1 * theta1 * x9 + F3 * theta3 * x9 -
            F4 * theta4 * x7 + F2 * theta2 * x7 +
            F_sum - p.m * p.g - p.C3 * x6
        )

        # Angular acceleration in roll (Eq. 34 from paper)
        dx8 = (1 / p.Ix) * (
            -p.l * p.C1_prime * x8 + p.M1 * theta1 - p.M3 * theta3 +
            p.M2_prime + p.M4_prime + tau1
        )

        # Angular acceleration in pitch (Eq. 36 from paper)
        dx10 = (1 / p.Iy) * (
            -p.l * p.C2_prime * x10 + p.M4 * theta4 - p.M2 * theta2 +
            p.M1_prime + p.M3_prime + tau2
        )

        # Angular acceleration in yaw (Eq. 38 from paper)
        dx12 = (1 / p.Iz) * (
            p.l * (F1 * theta1 + F2 * theta2 + F3 * theta3 + F4 * theta4) -
            p.l * p.C3_prime * x12 + p.M1 - p.M2 + p.M3 - p.M4
        )

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6,
                        dx7, dx8, dx9, dx10, dx11, dx12])

    def step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Integrate dynamics one timestep using Euler method.

        Args:
            x: Current state [12]
            u: Control input [6 or 7]
            dt: Timestep [s]

        Returns:
            np.ndarray: Next state [12]
        """
        dx_dt = self.dynamics(x, u)
        x_next = x + dx_dt * dt

        # Apply state constraints
        x_next[0] = np.clip(x_next[0], -50, 50)  # x position
        x_next[2] = np.clip(x_next[2], -50, 50)  # y position
        x_next[4] = np.clip(x_next[4], 0.5, 20)  # z position (above ground)
        x_next[6] = np.clip(x_next[6], -np.pi / 3, np.pi / 3)  # Roll
        x_next[8] = np.clip(x_next[8], -np.pi / 3, np.pi / 3)  # Pitch
        x_next[10] = np.clip(x_next[10], -np.pi, np.pi)  # Yaw

        return x_next

    def get_position(self, x: np.ndarray) -> np.ndarray:
        """Extract position from state vector.

        Args:
            x: State vector [12]

        Returns:
            np.ndarray: Position [x, y, z]
        """
        return np.array([x[0], x[2], x[4]])

    def get_velocity(self, x: np.ndarray) -> np.ndarray:
        """Extract velocity from state vector.

        Args:
            x: State vector [12]

        Returns:
            np.ndarray: Velocity [ẋ, ẏ, ż]
        """
        return np.array([x[1], x[3], x[5]])

    def get_attitude(self, x: np.ndarray) -> np.ndarray:
        """Extract Euler angles from state vector.

        Args:
            x: State vector [12]

        Returns:
            np.ndarray: Euler angles [φ, θ, ψ] (roll, pitch, yaw)
        """
        return np.array([x[6], x[8], x[10]])

    def get_angular_velocity(self, x: np.ndarray) -> np.ndarray:
        """Extract angular velocity from state vector.

        Args:
            x: State vector [12]

        Returns:
            np.ndarray: Angular velocity [φ̇, θ̇, ψ̇]
        """
        return np.array([x[7], x[9], x[11]])
