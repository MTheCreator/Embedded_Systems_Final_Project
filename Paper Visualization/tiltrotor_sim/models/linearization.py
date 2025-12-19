"""
System linearization using Taylor series expansion.
"""

import numpy as np
from .dynamics import TiltRotorUAV
from .parameters import TiltRotorUAVParameters


class SystemLinearization:
    """Linearization using Taylor series expansion following paper Eq. (43-53).

    This class computes the linearized system matrices A and B around
    an operating point (x*, u*) using the affine control system form:
    ẋ = f(x) + g(x)u

    The linearization is:
    A = ∇ₓf(x*) + Σᵢ∇ₓ[gᵢ(x*)]uᵢ*
    B = g(x*)

    Attributes:
        uav (TiltRotorUAV): Nonlinear UAV dynamics model
        params (TiltRotorUAVParameters): Physical parameters
    """

    def __init__(self, uav: TiltRotorUAV):
        """Initialize linearization module.

        Args:
            uav: Nonlinear UAV dynamics model
        """
        self.uav = uav
        self.params = uav.params

    def compute_jacobian_f(self, x_star: np.ndarray, u_star: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix ∇ₓf(x) as per Eq. 49.

        This computes the drift term (f(x)) Jacobian only,
        not including control-dependent terms.

        Args:
            x_star: Operating point state [12]
            u_star: Operating point control [6]

        Returns:
            np.ndarray: Jacobian matrix A_f [12×12]
        """
        p = self.params
        x11 = x_star[10]  # Yaw
        x7 = x_star[6]    # Roll
        x9 = x_star[8]    # Pitch

        c11 = np.cos(x11)
        s11 = np.sin(x11)

        F_sum = p.F1 + p.F2 + p.F3 + p.F4

        A_f = np.zeros((12, 12))

        # Position-velocity relationships (rows 0,2,4,6,8,10)
        A_f[0, 1] = 1.0
        A_f[2, 3] = 1.0
        A_f[4, 5] = 1.0
        A_f[6, 7] = 1.0
        A_f[8, 9] = 1.0
        A_f[10, 11] = 1.0

        # Damping terms (rows 1, 3, 5, 7, 9, 11)
        A_f[1, 1] = -p.C1 / p.m
        A_f[3, 3] = -p.C2 / p.m
        A_f[5, 5] = -p.C3 / p.m
        A_f[7, 7] = -p.l * p.C1_prime / p.Ix
        A_f[9, 9] = -p.l * p.C2_prime / p.Iy
        A_f[11, 11] = -p.l * p.C3_prime / p.Iz

        # Coupling terms for ẍ (row 1) - from Eq. 49
        A_f[1, 6] = F_sum / p.m * s11
        A_f[1, 8] = F_sum / p.m * c11
        q1 = s11 * x7 + c11 * x9
        A_f[1, 10] = F_sum / p.m * q1

        # Coupling terms for ÿ (row 3) - from Eq. 49
        A_f[3, 6] = -F_sum / p.m * c11
        A_f[3, 8] = F_sum / p.m * s11
        q2 = -c11 * x7 + s11 * x9
        A_f[3, 10] = F_sum / p.m * q2

        return A_f

    def compute_jacobian_g_contributions(self, x_star: np.ndarray, u_star: np.ndarray) -> np.ndarray:
        """Compute ∇ₓ[gᵢ(x)]uᵢ contributions as per Eq. 44.

        This properly computes state-dependent control terms.

        Args:
            x_star: Operating point state [12]
            u_star: Operating point control [6]

        Returns:
            np.ndarray: Jacobian matrix A_g [12×12]
        """
        p = self.params
        x11 = x_star[10]
        x7 = x_star[6]
        x9 = x_star[8]

        c11 = np.cos(x11)
        s11 = np.sin(x11)

        u1, u2, u3, u4, tau1, tau2 = u_star

        A_g = np.zeros((12, 12))

        # Contribution from θ₁ (u1) - from Eq. 50
        A_g[1, 10] = -(1 / p.m) * p.F1 * s11 * u1
        A_g[3, 10] = (1 / p.m) * p.F1 * c11 * u1
        A_g[5, 8] = -(p.F1 / p.m) * u1

        # Contribution from θ₂ (u2) - from Eq. 51
        A_g[1, 6] = (1 / p.m) * p.F2 * c11 * x9 * u2
        A_g[1, 8] = (1 / p.m) * p.F2 * c11 * x7 * u2
        A_g[1, 10] = (1 / p.m) * p.F2 * (-c11 + s11 * x7 * x9) * u2

        A_g[3, 6] = (1 / p.m) * p.F2 * s11 * x9 * u2
        A_g[3, 8] = (1 / p.m) * p.F2 * s11 * x7 * u2
        A_g[3, 10] = (1 / p.m) * p.F2 * (s11 + c11 * x7 * x9) * u2

        A_g[5, 6] = (p.F2 / p.m) * u2

        # Contribution from θ₃ (u3) - from Eq. 52
        A_g[1, 10] += (1 / p.m) * p.F3 * s11 * u3
        A_g[3, 10] += -(1 / p.m) * p.F3 * c11 * u3
        A_g[5, 8] += (p.F3 / p.m) * u3

        # Contribution from θ₄ (u4) - from Eq. 53
        A_g[1, 6] += (1 / p.m) * p.F4 * c11 * x9 * u4
        A_g[1, 8] += (1 / p.m) * p.F4 * c11 * x7 * u4
        A_g[1, 10] += (1 / p.m) * p.F4 * (c11 + s11 * x7 * x9) * u4

        A_g[3, 6] += -(1 / p.m) * p.F4 * s11 * x9 * u4
        A_g[3, 8] += -(1 / p.m) * p.F4 * s11 * x7 * u4
        A_g[3, 10] += (1 / p.m) * p.F4 * (-s11 - c11 * x7 * x9) * u4

        A_g[5, 6] += -(p.F4 / p.m) * u4

        # τ₁ and τ₂ don't contribute to ∇ₓg terms (constant gain)

        return A_g

    def compute_B_matrix(self, x_star: np.ndarray) -> np.ndarray:
        """Compute control input gain matrix B = g(x*) from Eq. 39.

        This is evaluated at x*, not including u*.

        Args:
            x_star: Operating point state [12]

        Returns:
            np.ndarray: Control matrix B [12×6]
        """
        p = self.params
        x11 = x_star[10]
        x7 = x_star[6]
        x9 = x_star[8]

        c11 = np.cos(x11)
        s11 = np.sin(x11)

        B = np.zeros((12, 6))

        # Row 1 (ẍ) - Eq. (28) control terms
        B[1, 0] = (1 / p.m) * p.F1 * c11
        B[1, 1] = (1 / p.m) * (-p.F2 * s11 + p.F2 * c11 * x7 * x9)
        B[1, 2] = -(1 / p.m) * p.F3 * c11
        B[1, 3] = (1 / p.m) * (p.F4 * s11 + p.F4 * c11 * x7 * x9)

        # Row 3 (ÿ) - Eq. (30) control terms
        B[3, 0] = (1 / p.m) * p.F1 * s11
        B[3, 1] = (1 / p.m) * (p.F2 * c11 + p.F2 * s11 * x7 * x9)
        B[3, 2] = -(1 / p.m) * p.F3 * s11
        B[3, 3] = (1 / p.m) * (-p.F4 * c11 - p.F4 * s11 * x7 * x9)

        # Row 5 (z̈) - Eq. (32) control terms
        B[5, 0] = -(1 / p.m) * p.F1 * x9
        B[5, 1] = (1 / p.m) * p.F2 * x7
        B[5, 2] = (1 / p.m) * p.F3 * x9
        B[5, 3] = -(1 / p.m) * p.F4 * x7

        # Row 7 (φ̈) - Eq. (34) control terms
        B[7, 0] = p.M1 / p.Ix
        B[7, 2] = -p.M3 / p.Ix
        B[7, 4] = 1 / p.Ix

        # Row 9 (θ̈) - Eq. (36) control terms
        B[9, 1] = -p.M2 / p.Iy
        B[9, 3] = p.M4 / p.Iy
        B[9, 5] = 1 / p.Iy

        # Row 11 (ψ̈) - Eq. (38) control terms
        B[11, 0] = p.l * p.F1 / p.Iz
        B[11, 1] = p.l * p.F2 / p.Iz
        B[11, 2] = p.l * p.F3 / p.Iz
        B[11, 3] = p.l * p.F4 / p.Iz

        return B

    def linearize(self, x_star: np.ndarray, u_star: np.ndarray) -> tuple:
        """Linearize system around operating point (x*, u*) as per Eq. 44-45.

        The linearized system is: δẋ = A δx + B δu
        where:
            A = ∇ₓf(x*) + Σᵢ∇ₓ[gᵢ(x*)]uᵢ*
            B = g(x*)

        Args:
            x_star: Operating point state [12]
            u_star: Operating point control [6]

        Returns:
            tuple: (A, B) where A is [12×12] and B is [12×6]
        """
        # Ensure inputs are bounded
        u_star = np.clip(u_star, [-0.3, -0.3, -0.3, -0.3, -2.0, -2.0],
                         [0.3, 0.3, 0.3, 0.3, 2.0, 2.0])

        # A = ∇ₓf(x) + Σ∇ₓ[gᵢ(x)]uᵢ
        A_f = self.compute_jacobian_f(x_star, u_star)
        A_g = self.compute_jacobian_g_contributions(x_star, u_star)
        A = A_f + A_g

        # B = g(x*)
        B = self.compute_B_matrix(x_star)

        return A, B
