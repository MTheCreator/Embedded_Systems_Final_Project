"""
PyBullet-based physics simulator for advanced simulations.
"""

import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import List, Optional
from ..models.parameters import TiltRotorUAVParameters
from ..navigation.obstacles import Obstacle


class PyBulletSimulator:
    """PyBullet physics simulator for tilt-rotor UAV.

    This class integrates PyBullet for realistic physics simulation including:
    - Rigid body dynamics
    - Collision detection
    - Realistic rendering
    - Force/torque application

    Attributes:
        params (TiltRotorUAVParameters): UAV parameters
        client_id (int): PyBullet client ID
        uav_id (int): UAV body ID in PyBullet
        dt (float): Physics timestep
        gui (bool): Whether GUI is enabled
    """

    def __init__(self, params: TiltRotorUAVParameters, dt: float = 0.01,
                 gui: bool = True, real_time: bool = True):
        """Initialize PyBullet simulator.

        Args:
            params: UAV physical parameters
            dt: Physics simulation timestep [s]
            gui: Whether to show GUI
            real_time: Whether to run in real-time (adds delays for visualization)
        """
        self.params = params
        self.dt = dt
        self.gui = gui
        self.real_time = real_time
        self.last_step_time = time.time()

        # Initialize PyBullet
        if gui:
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -params.g)
        p.setTimeStep(dt)

        # Enable real-time simulation for better visualization
        if gui and real_time:
            p.setRealTimeSimulation(0)  # We'll manually control timing

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create UAV body
        self.uav_id = None
        self._create_uav()

        # Obstacle IDs
        self.obstacle_ids = []

        # Camera tracking
        self.follow_camera = True

        # Trajectory visualization
        self.trajectory_points = []
        self.draw_trajectory = True

    def _create_uav(self):
        """Create UAV body in PyBullet."""
        # Create collision shape (box approximation)
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.1]
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.1],
            rgbaColor=[0.2, 0.5, 0.8, 1.0]
        )

        # Create multi-body with rotors as separate visual elements
        self.uav_id = p.createMultiBody(
            baseMass=self.params.m,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 1],
            baseOrientation=[0, 0, 0, 1]
        )

        # Set inertia
        p.changeDynamics(
            self.uav_id, -1,
            localInertiaDiagonal=[self.params.Ix, self.params.Iy, self.params.Iz]
        )

        # Add visual markers for rotors
        rotor_positions = [
            [self.params.l, 0, 0],
            [0, self.params.l, 0],
            [-self.params.l, 0, 0],
            [0, -self.params.l, 0]
        ]

        for i, pos in enumerate(rotor_positions):
            rotor_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.08,
                length=0.02,
                rgbaColor=[0.8, 0.1, 0.1, 1.0]
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=rotor_visual,
                basePosition=pos
            )

    def reset(self, initial_state: np.ndarray):
        """Reset simulation to initial state.

        Args:
            initial_state: Initial state [x, ẋ, y, ẏ, z, ż, φ, φ̇, θ, θ̇, ψ, ψ̇]
        """
        position = [initial_state[0], initial_state[2], initial_state[4]]

        # Convert Euler angles to quaternion
        roll, pitch, yaw = initial_state[6], initial_state[8], initial_state[10]
        orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

        p.resetBasePositionAndOrientation(self.uav_id, position, orientation)

        # Set velocities
        linear_vel = [initial_state[1], initial_state[3], initial_state[5]]
        angular_vel = [initial_state[7], initial_state[9], initial_state[11]]
        p.resetBaseVelocity(self.uav_id, linear_vel, angular_vel)

    def step(self, control: np.ndarray) -> np.ndarray:
        """Step simulation with control input.

        Args:
            control: Control vector [θ₁, θ₂, θ₃, θ₄, τ₁, τ₂, δ_thrust]

        Returns:
            np.ndarray: New state [12]
        """
        # Extract control
        if len(control) == 6:
            theta1, theta2, theta3, theta4, tau1, tau2 = control
            delta_thrust = 0.0
        else:
            theta1, theta2, theta3, theta4, tau1, tau2, delta_thrust = control

        # Compute forces and torques from tilt-rotor model
        thrust_scale = 1.0 + delta_thrust
        F1 = self.params.F1 * thrust_scale
        F2 = self.params.F2 * thrust_scale
        F3 = self.params.F3 * thrust_scale
        F4 = self.params.F4 * thrust_scale

        # Get current orientation
        _, orientation = p.getBasePositionAndOrientation(self.uav_id)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        # Compute rotor forces in body frame
        rotor_positions = [
            np.array([self.params.l, 0, 0]),
            np.array([0, self.params.l, 0]),
            np.array([-self.params.l, 0, 0]),
            np.array([0, -self.params.l, 0])
        ]

        forces = [F1, F2, F3, F4]
        tilts = [theta1, theta2, theta3, theta4]

        # Apply forces and torques
        total_force_body = np.zeros(3)
        total_torque_body = np.zeros(3)

        for i, (pos, force, tilt) in enumerate(zip(rotor_positions, forces, tilts)):
            # Force direction in body frame (tilted)
            force_direction = np.array([
                np.sin(tilt),
                0,
                np.cos(tilt)
            ])
            force_vec = force * force_direction

            total_force_body += force_vec
            total_torque_body += np.cross(pos, force_vec)

        # Add auxiliary torques
        total_torque_body[0] += tau1  # Roll torque
        total_torque_body[1] += tau2  # Pitch torque

        # Add drag
        linear_vel, angular_vel = p.getBaseVelocity(self.uav_id)
        drag_force = -np.array([
            self.params.C1 * linear_vel[0],
            self.params.C2 * linear_vel[1],
            self.params.C3 * linear_vel[2]
        ])
        total_force_body += rotation_matrix.T @ drag_force

        # Transform to world frame and apply
        total_force_world = rotation_matrix @ total_force_body
        total_torque_world = rotation_matrix @ total_torque_body

        p.applyExternalForce(
            self.uav_id, -1,
            total_force_world, [0, 0, 0],
            p.WORLD_FRAME
        )

        p.applyExternalTorque(
            self.uav_id, -1,
            total_torque_world,
            p.WORLD_FRAME
        )

        # Step simulation
        p.stepSimulation()

        # Real-time delay for visualization
        if self.gui and self.real_time:
            current_time = time.time()
            elapsed = current_time - self.last_step_time
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.last_step_time = time.time()

        # Update camera to follow drone
        if self.gui and self.follow_camera:
            position, _ = p.getBasePositionAndOrientation(self.uav_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=15.0,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=position
            )

            # Draw trajectory line
            if self.draw_trajectory:
                if len(self.trajectory_points) > 0:
                    prev_point = self.trajectory_points[-1]
                    # Only add point if drone moved significantly
                    if np.linalg.norm(np.array(position) - np.array(prev_point)) > 0.1:
                        p.addUserDebugLine(prev_point, position, [0, 0, 1], 2, lifeTime=0)
                        self.trajectory_points.append(position)
                else:
                    self.trajectory_points.append(position)

        # Get new state
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get current UAV state.

        Returns:
            np.ndarray: State [x, ẋ, y, ẏ, z, ż, φ, φ̇, θ, θ̇, ψ, ψ̇]
        """
        position, orientation = p.getBasePositionAndOrientation(self.uav_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.uav_id)

        # Convert quaternion to Euler angles
        euler = p.getEulerFromQuaternion(orientation)

        state = np.array([
            position[0], linear_vel[0],
            position[1], linear_vel[1],
            position[2], linear_vel[2],
            euler[0], angular_vel[0],  # Roll
            euler[1], angular_vel[1],  # Pitch
            euler[2], angular_vel[2]   # Yaw
        ])

        return state

    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle to simulation.

        Args:
            obstacle: Obstacle object to add
        """
        obs_dict = obstacle.get_dict()

        if obs_dict['type'] == 'sphere':
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=obs_dict['radius']
            )
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=obs_dict['radius'],
                rgbaColor=[1.0, 0.2, 0.2, 0.7]
            )
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=obs_dict['center']
            )

        elif obs_dict['type'] == 'box':
            size = (obs_dict['max'] - obs_dict['min']) / 2
            center = (obs_dict['max'] + obs_dict['min']) / 2

            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=size
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=size,
                rgbaColor=[1.0, 0.5, 0.2, 0.7]
            )
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=center
            )

        self.obstacle_ids.append(obs_id)

    def add_waypoint_marker(self, position: np.ndarray, index: int):
        """Add visual marker for waypoint.

        Args:
            position: Waypoint position
            index: Waypoint index
        """
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.3,
            rgbaColor=[0.2, 1.0, 0.2, 0.8]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )

    def check_collision(self) -> bool:
        """Check if UAV is in collision with any obstacle.

        Returns:
            bool: True if collision detected
        """
        for obs_id in self.obstacle_ids:
            contact_points = p.getContactPoints(self.uav_id, obs_id)
            if contact_points is not None and len(contact_points) > 0:
                return True
        return False

    def set_camera(self, distance: float = 10.0, yaw: float = 45.0,
                   pitch: float = -30.0, target: np.ndarray = None):
        """Set camera view.

        Args:
            distance: Camera distance
            yaw: Camera yaw angle [degrees]
            pitch: Camera pitch angle [degrees]
            target: Camera target position
        """
        if self.gui:
            if target is None:
                target = [0, 0, 5]
            p.resetDebugVisualizerCamera(distance, yaw, pitch, target)

    def enable_camera_follow(self, enable: bool = True):
        """Enable or disable camera following the drone.

        Args:
            enable: True to follow drone, False for manual camera
        """
        self.follow_camera = enable

    def enable_trajectory_drawing(self, enable: bool = True):
        """Enable or disable trajectory line drawing.

        Args:
            enable: True to draw trajectory, False to disable
        """
        self.draw_trajectory = enable

    def add_debug_text(self, text: str, position: np.ndarray, color: list = [1, 1, 1]):
        """Add debug text to visualization.

        Args:
            text: Text to display
            position: 3D position for text
            color: RGB color [0-1]
        """
        if self.gui:
            p.addUserDebugText(text, position, textColorRGB=color, textSize=1.5, lifeTime=0)

    def close(self):
        """Close PyBullet connection."""
        p.disconnect(self.client_id)
