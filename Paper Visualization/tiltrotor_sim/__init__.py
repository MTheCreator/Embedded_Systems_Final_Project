"""
Tiltrotor UAV Simulation Package

A modular simulation framework for tilt-rotor UAVs with:
- H-infinity robust control
- Kalman filtering for state estimation
- PyBullet physics engine integration
- Waypoint navigation and obstacle avoidance
"""

__version__ = "2.0.0"
__author__ = "UAV Simulation Team"

from .models.parameters import TiltRotorUAVParameters
from .models.dynamics import TiltRotorUAV
from .control.hinf_controller import HInfinityController
from .control.kalman_filter import HInfinityKalmanFilter
from .navigation.guidance import WaypointGuidanceController, NavigationState

__all__ = [
    'TiltRotorUAVParameters',
    'TiltRotorUAV',
    'HInfinityController',
    'HInfinityKalmanFilter',
    'WaypointGuidanceController',
    'NavigationState',
]
