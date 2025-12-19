"""
H-infinity Kalman filter for state estimation.
"""

import numpy as np


class HInfinityKalmanFilter:
    """H-infinity Kalman Filter as per Eq. 75-76.

    This filter provides robust state estimation with guaranteed
    performance bound under worst-case disturbances.

    The filter equations are:
    Prediction: x̂ₖ₊₁|ₖ = Ax̂ₖ|ₖ + Buₖ
    Update: x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + K(yₖ - Cx̂ₖ|ₖ₋₁)

    Attributes:
        n (int): Number of states
        m (int): Number of measurements
        theta (float): H-infinity performance parameter
        x_hat (np.ndarray): State estimate [n]
        P_minus (np.ndarray): Error covariance [n×n]
        Q (np.ndarray): Process noise covariance [n×n]
        R (np.ndarray): Measurement noise covariance [m×m]
        C (np.ndarray): Measurement matrix [m×n]
        W (np.ndarray): Disturbance weighting [n×n]
    """

    def __init__(self, n_states: int = 12, n_measurements: int = 6,
                 theta: float = 0.001):
        """Initialize H-infinity Kalman filter.

        Args:
            n_states: Number of state variables
            n_measurements: Number of measurements
            theta: H-infinity performance parameter (smaller = better performance)
        """
        self.n = n_states
        self.m = n_measurements
        self.theta = theta

        # State estimate and covariance
        self.x_hat = np.zeros(n_states)
        self.P_minus = 0.1 * np.eye(n_states)

        # Noise covariances
        self.Q = 0.001 * np.eye(n_states)  # Process noise
        self.R = 0.01 * np.eye(n_measurements)  # Measurement noise

        # Measurement matrix C: measure x, y, z, φ, θ, ψ
        self.C = np.zeros((n_measurements, n_states))
        self.C[0, 0] = 1.0   # x position
        self.C[1, 2] = 1.0   # y position
        self.C[2, 4] = 1.0   # z position
        self.C[3, 6] = 1.0   # roll
        self.C[4, 8] = 1.0   # pitch
        self.C[5, 10] = 1.0  # yaw

        # Disturbance weighting
        self.W = np.eye(n_states)

    def predict(self, A: np.ndarray, B: np.ndarray, u: np.ndarray, dt: float):
        """Time update (prediction step) as per Eq. 76.

        Args:
            A: State transition matrix [n×n]
            B: Control matrix [n×m]
            u: Control input [m]
            dt: Timestep [s]
        """
        # Discretize system matrices
        A_d = np.eye(self.n) + A * dt
        B_d = B * dt

        # Prediction: x̂ₖ₊₁|ₖ = Ax̂ₖ|ₖ + Buₖ
        self.x_hat = A_d @ self.x_hat + B_d @ u

        # Covariance prediction with H-infinity term
        try:
            P_inv = np.linalg.inv(self.P_minus + 1e-6 * np.eye(self.n))
            D_inv = P_inv - self.theta * self.W + self.C.T @ np.linalg.inv(self.R) @ self.C
            D = np.linalg.inv(D_inv + 1e-6 * np.eye(self.n))
        except np.linalg.LinAlgError:
            D = self.P_minus

        self.P_minus = A_d @ self.P_minus @ D @ A_d.T + self.Q
        # Ensure symmetry
        self.P_minus = (self.P_minus + self.P_minus.T) / 2

    def update(self, y: np.ndarray):
        """Measurement update (correction step) as per Eq. 75.

        Args:
            y: Measurement vector [m]
        """
        try:
            # Innovation: ỹ = y - Cx̂
            innovation = y - self.C @ self.x_hat

            # Innovation covariance: S = CP⁻C^T + R
            S = self.C @ self.P_minus @ self.C.T + self.R

            # Kalman gain: K = P⁻C^T S⁻¹
            K = self.P_minus @ self.C.T @ np.linalg.inv(S)

            # State update: x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kỹ
            self.x_hat = self.x_hat + K @ innovation

            # Covariance update: Pₖ|ₖ = (I - KC)Pₖ|ₖ₋₁
            self.P_minus = (np.eye(self.n) - K @ self.C) @ self.P_minus
            # Ensure symmetry
            self.P_minus = (self.P_minus + self.P_minus.T) / 2

        except np.linalg.LinAlgError:
            # If update fails, keep prediction
            pass

    def get_estimate(self) -> np.ndarray:
        """Get current state estimate.

        Returns:
            np.ndarray: State estimate [n]
        """
        return self.x_hat.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current error covariance.

        Returns:
            np.ndarray: Error covariance matrix [n×n]
        """
        return self.P_minus.copy()

    def reset(self, x0: np.ndarray = None):
        """Reset filter to initial state.

        Args:
            x0: Initial state estimate (default: zeros)
        """
        if x0 is not None:
            self.x_hat = x0.copy()
        else:
            self.x_hat = np.zeros(self.n)
        self.P_minus = 0.1 * np.eye(self.n)

    def set_process_noise(self, Q: np.ndarray):
        """Set process noise covariance.

        Args:
            Q: Process noise covariance [n×n]
        """
        assert Q.shape == (self.n, self.n), "Q must be n×n"
        self.Q = Q

    def set_measurement_noise(self, R: np.ndarray):
        """Set measurement noise covariance.

        Args:
            R: Measurement noise covariance [m×m]
        """
        assert R.shape == (self.m, self.m), "R must be m×m"
        self.R = R

    def set_measurement_matrix(self, C: np.ndarray):
        """Set measurement matrix.

        Args:
            C: Measurement matrix [m×n]
        """
        assert C.shape == (self.m, self.n), f"C must be {self.m}×{self.n}"
        self.C = C
