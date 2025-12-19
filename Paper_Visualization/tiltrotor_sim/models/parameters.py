"""
Physical parameters for tilt-rotor UAV.
"""

import numpy as np


class TiltRotorUAVParameters:
    """Physical parameters for a quadrotor UAV with tilting rotors.

    This class encapsulates all physical constants and design parameters
    for the tilt-rotor UAV including mass, inertia, aerodynamic coefficients,
    and rotor characteristics.

    Attributes:
        m (float): Total mass of the UAV [kg]
        g (float): Gravitational acceleration [m/s²]
        Ix, Iy, Iz (float): Moments of inertia [kg⋅m²]
        kf (float): Thrust coefficient [N⋅s²]
        km (float): Moment coefficient [N⋅m⋅s²]
        l (float): Arm length [m]
        C1, C2, C3 (float): Linear damping coefficients [N⋅s/m]
        C1_prime, C2_prime, C3_prime (float): Angular damping coefficients [N⋅m⋅s]
        n_states (int): Number of state variables (12)
        n_inputs (int): Number of control inputs (6)
    """

    def __init__(self):
        """Initialize UAV parameters with default values."""
        # Mass properties
        self.m = 2.0  # Total mass [kg]
        self.g = 9.81  # Gravitational acceleration [m/s²]

        # Moments of inertia [kg⋅m²]
        self.Ix = 0.0347563
        self.Iy = 0.0458929
        self.Iz = 0.0977

        # Rotor characteristics
        self.kf = 1.0e-5  # Thrust coefficient [N⋅s²]
        self.km = 1.0e-7  # Moment coefficient [N⋅m⋅s²]
        self.l = 0.25     # Arm length [m]

        # Aerodynamic damping coefficients
        self.C1 = 0.1        # Linear damping x [N⋅s/m]
        self.C2 = 0.1        # Linear damping y [N⋅s/m]
        self.C3 = 0.1        # Linear damping z [N⋅s/m]
        self.C1_prime = 0.01 # Angular damping roll [N⋅m⋅s]
        self.C2_prime = 0.01 # Angular damping pitch [N⋅m⋅s]
        self.C3_prime = 0.01 # Angular damping yaw [N⋅m⋅s]

        # Additional moment terms (coupling effects)
        self.M1_prime = 0.0
        self.M2_prime = 0.0
        self.M3_prime = 0.0
        self.M4_prime = 0.0

        # State and input dimensions
        self.n_states = 12  # [x, ẋ, y, ẏ, z, ż, φ, φ̇, θ, θ̇, ψ, ψ̇]
        self.n_inputs = 6   # [θ₁, θ₂, θ₃, θ₄, τ₁, τ₂]

        # Nominal rotor speeds and forces (hovering condition)
        omega_hover = np.sqrt(self.m * self.g / (4 * self.kf))
        self.omega1 = omega_hover
        self.omega2 = omega_hover
        self.omega3 = omega_hover
        self.omega4 = omega_hover

        # Nominal thrust forces [N]
        self.F1 = self.kf * self.omega1**2
        self.F2 = self.kf * self.omega2**2
        self.F3 = self.kf * self.omega3**2
        self.F4 = self.kf * self.omega4**2

        # Nominal moments [N⋅m]
        self.M1 = self.km * self.omega1**2
        self.M2 = self.km * self.omega2**2
        self.M3 = self.km * self.omega3**2
        self.M4 = self.km * self.omega4**2

    def get_total_hover_thrust(self):
        """Calculate total thrust force at hover.

        Returns:
            float: Total thrust force [N]
        """
        return self.F1 + self.F2 + self.F3 + self.F4

    def get_inertia_matrix(self):
        """Get the inertia tensor as a diagonal matrix.

        Returns:
            np.ndarray: 3x3 inertia matrix [kg⋅m²]
        """
        return np.diag([self.Ix, self.Iy, self.Iz])

    def __repr__(self):
        """String representation of parameters."""
        return (f"TiltRotorUAVParameters(m={self.m}kg, "
                f"Hover_Thrust={self.get_total_hover_thrust():.2f}N)")
