"""
H-infinity robust controller for UAV stabilization.
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class HInfinityController:
    """H-infinity stabilizing feedback controller as per Eq. 60-61.

    This controller solves the H∞ control problem with disturbance
    attenuation level γ = 1/ρ. The control law is:
    u = -Kx where K = (1/r)B^T P

    and P solves the algebraic Riccati equation (ARE):
    A^T P + PA + Q - P(2/r BB^T - 1/ρ² LL^T)P = 0

    Attributes:
        r (float): Control weighting parameter
        rho (float): Disturbance attenuation parameter (γ = 1/ρ)
        P (np.ndarray): Solution to Riccati equation
        K (np.ndarray): Feedback gain matrix
        riccati_solved (bool): Whether ARE was successfully solved
    """

    def __init__(self, r: float = 1.0, rho: float = 0.1):
        """Initialize H-infinity controller.

        Args:
            r: Control effort weight (larger = less aggressive control)
            rho: Disturbance attenuation (smaller = better disturbance rejection)
        """
        self.r = r
        self.rho = rho
        self.P = None
        self.K = None
        self.riccati_solved = False

    def solve_riccati(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Solve algebraic Riccati equation from Eq. 61.

        Paper equation: A^T P + PA + Q - P(2/r BB^T - 1/ρ² LL^T)P = 0

        Standard ARE: A^T P + PA + Q - PBR^{-1}B^T P = 0

        We match by setting: B_eff = [√(2/r)B, √(1/ρ²)I]
                            R_eff = I

        Args:
            A: State matrix [n×n]
            B: Control matrix [n×m]
            Q: State weighting matrix [n×n]

        Returns:
            np.ndarray: Solution P [n×n]
        """
        n = A.shape[0]
        m = B.shape[1]

        # Add regularization for numerical stability
        A_reg = A + 1e-6 * np.eye(n)
        Q_reg = Q + 1e-6 * np.eye(n)

        # Check observability proxy (Q should penalize observable states)
        if np.linalg.matrix_rank(Q_reg) < n:
            print("Warning: Q is not full rank, adding regularization")
            Q_reg = Q_reg + 0.01 * np.eye(n)

        try:
            # Extended B matrix approach
            # B_extended = [sqrt(2/r)*B, sqrt(1/rho^2)*I]
            B_extended = np.hstack([
                np.sqrt(2.0 / self.r) * B,
                np.sqrt(1.0 / self.rho**2) * np.eye(n)
            ])

            R_extended = np.eye(m + n)

            # Solve extended ARE
            self.P = solve_continuous_are(A_reg, B_extended, Q_reg, R_extended)

            # Verify solution
            eigvals = np.linalg.eigvals(self.P)
            if np.all(eigvals > -1e-10):  # Allow small negative due to numerical errors
                self.riccati_solved = True
                # Force P to be symmetric
                self.P = (self.P + self.P.T) / 2
            else:
                print(f"Warning: P has negative eigenvalues: {np.min(eigvals)}")
                self.riccati_solved = False
                self.P = Q_reg + np.eye(n)

        except Exception as e:
            print(f"Warning: Riccati solution failed: {e}")
            self.riccati_solved = False
            self.P = Q + 10 * np.eye(n)

        return self.P

    def compute_gain(self, B: np.ndarray) -> np.ndarray:
        """Compute feedback gain: K = (1/r)B^T P from Eq. 60.

        Args:
            B: Control matrix [n×m]

        Returns:
            np.ndarray: Feedback gain K [m×n]
        """
        if not self.riccati_solved:
            print("Warning: Using fallback gain")
            m = B.shape[1]
            n = B.shape[0]
            self.K = 0.1 * np.random.randn(m, n) * 0.01
            return self.K

        self.K = (1 / self.r) * B.T @ self.P
        return self.K

    def control_law(self, error: np.ndarray, altitude_error: float = 0.0,
                   dt: float = 0.01) -> np.ndarray:
        """Compute control with gentle thrust modulation.

        Args:
            error: State error vector [n]
            altitude_error: Altitude tracking error [m]
            dt: Timestep [s]

        Returns:
            np.ndarray: Control vector [7] = [θ₁, θ₂, θ₃, θ₄, τ₁, τ₂, δ_thrust]
        """
        if self.K is None:
            return np.zeros(7)

        # Basic 6-DOF control: u = -Kx
        u_basic = -self.K @ error
        u_basic[0:4] = np.clip(u_basic[0:4], -0.2, 0.2)  # Tilt angles
        u_basic[4:6] = np.clip(u_basic[4:6], -1.5, 1.5)  # Auxiliary torques

        # Gentle thrust adjustment for large altitude errors
        if abs(altitude_error) > 1.0:
            delta_thrust = -0.0025 * altitude_error  # Very small gain
        else:
            delta_thrust = 0.0  # Don't interfere with small errors

        delta_thrust = np.clip(delta_thrust, -0.15, 0.15)

        # Combine into 7-element control vector
        u = np.hstack([u_basic, delta_thrust])

        return u

    def get_closed_loop_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Get closed-loop system matrix A_cl = A - BK.

        Args:
            A: Open-loop state matrix [n×n]
            B: Control matrix [n×m]

        Returns:
            np.ndarray: Closed-loop matrix [n×n]
        """
        if self.K is None:
            return A
        return A - B @ self.K

    def is_stable(self, A: np.ndarray, B: np.ndarray) -> bool:
        """Check if closed-loop system is stable.

        Args:
            A: Open-loop state matrix
            B: Control matrix

        Returns:
            bool: True if all eigenvalues have negative real parts
        """
        A_cl = self.get_closed_loop_matrix(A, B)
        eigvals = np.linalg.eigvals(A_cl)
        return np.all(np.real(eigvals) < 0)
