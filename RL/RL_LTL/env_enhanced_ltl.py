"""
Realistic quadcopter physics with LTL specifications
Fixed rotation math, proper inertia, better drag model
"""
import numpy as np
import pygame
from collections import deque

class Obstacle:
    """
    3D obstacle - can be sphere or box
    Used for collision checking and distance queries
    """
    def __init__(self, position, size, shape='box'):
        self.position = np.array(position, dtype=float)
        self.size = size
        self.shape = shape
    
    def check_collision(self, point, safety_margin=0.3):
        """Check if point collides with obstacle (with safety margin)"""
        if self.shape == 'sphere':
            distance = np.linalg.norm(point - self.position)
            return distance < (self.size + safety_margin)
        elif self.shape == 'box':
            half_size = np.array(self.size) / 2 + safety_margin
            diff = np.abs(point - self.position)
            return np.all(diff < half_size)
        return False
    
    def distance_to(self, point):
        """Calculate minimum distance from point to obstacle surface"""
        if self.shape == 'sphere':
            return max(0, np.linalg.norm(point - self.position) - self.size)
        elif self.shape == 'box':
            half_size = np.array(self.size) / 2
            diff = np.abs(point - self.position) - half_size
            # If inside box, distance is 0
            # If outside, compute distance to nearest surface
            outside_distance = np.maximum(diff, 0)
            return np.linalg.norm(outside_distance)

class RealisticDroneEnvLTL:
    """
    Quadcopter simulation with proper physics
    
    Motor layout (X-configuration):
        0: Front-Right (CW)
        1: Front-Left  (CCW)
        2: Back-Left   (CW)
        3: Back-Right  (CCW)
    
    Coordinate system: X=forward, Y=left, Z=up (NED-ish but Z-up)
    """
    def __init__(self, num_obstacles=5, fixed_obstacles=True, fixed_target=True, ltl_monitor=None):
        # Physical parameters (roughly based on real quadcopter)
        self.mass = 1.0  # kg
        self.gravity = 9.81  # m/s²
        self.dt = 0.02  # 50 Hz simulation
        
        # Drag coefficients
        self.linear_drag_coeff = 0.1  # simplified linear drag
        self.angular_drag = 0.05
        
        # Quadcopter geometry
        self.arm_length = 0.25  # meters from center to motor
        self.max_thrust_per_motor = 4.0  # Newtons (total ~16N vs weight ~10N = thrust-to-weight ~1.6)
        
        # Inertia tensor (kg*m²) - different for each axis
        # For X-config quad: roll/pitch similar, yaw higher (longer moment arm)
        self.inertia = np.array([
            [0.01, 0, 0],      # Roll inertia (Ixx)
            [0, 0.01, 0],      # Pitch inertia (Iyy)
            [0, 0, 0.02]       # Yaw inertia (Izz) - higher because mass is further from axis
        ])
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # Environment bounds
        self.max_pos = 10.0
        self.max_vel = 20.0
        self.max_angle = np.pi
        
        # Episode limits
        self.max_steps = 2000
        self.current_step = 0
        
        # Obstacles
        self.num_obstacles = num_obstacles
        self.fixed_obstacles = fixed_obstacles
        self.obstacles = []
        
        # Target
        self.fixed_target = fixed_target
        self.fixed_target_position = np.array([7.0, 3.0, 3.0])
        
        # Hovering requirements
        self.hover_zone_radius = 0.3  # must be within 30cm
        self.max_hover_velocity = 0.3  # m/s
        self.max_hover_tilt = 0.15  # radians (~8.6 degrees)
        self.hover_time_required = 100  # timesteps
        self.hover_counter = 0
        self.reached_target = False
        
        # LTL Monitor (if provided)
        self.ltl_monitor = ltl_monitor
        self.ltl_reward_history = deque(maxlen=100)
        self.ltl_violations_history = []
        
        # Setup obstacles
        if self.fixed_obstacles:
            self._generate_fixed_obstacles()
        
        # Pygame rendering setup
        pygame.init()
        self.screen_width = 1400
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("🚁 Drone RL with LTL Specifications (Fixed Physics)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 16)
        
        self.trail = deque(maxlen=50)
        self.motor_thrusts = np.zeros(4)
        
        self.reset()
    
    def _generate_fixed_obstacles(self):
        """Fixed obstacle course for consistent training"""
        self.obstacles = [
            Obstacle([-6, 3.0, 2.0], 0.7, 'sphere'),
            Obstacle([2, 0, 2.5], [1.5, 1.5, 3.0], 'box'),
            Obstacle([5, -2, 1.5], 0.6, 'sphere'),
            Obstacle([-2, -4, 2.0], [1.0, 1.0, 2.5], 'box'),
            Obstacle([7, 1, 3.5], 0.5, 'sphere'),
        ]
        self.obstacles = self.obstacles[:self.num_obstacles]
    
    def _generate_random_obstacles(self):
        """Generate random obstacles for variety"""
        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = [
                np.random.uniform(-7, 7),
                np.random.uniform(-7, 7),
                np.random.uniform(0.5, 4)
            ]
            if np.random.rand() < 0.5:
                # Sphere
                size = np.random.uniform(0.3, 0.8)
                self.obstacles.append(Obstacle(pos, size, 'sphere'))
            else:
                # Box
                size = [
                    np.random.uniform(0.5, 1.5),
                    np.random.uniform(0.5, 1.5),
                    np.random.uniform(1.0, 3.0)
                ]
                self.obstacles.append(Obstacle(pos, size, 'box'))
    
    def reset(self):
        """Reset environment to initial state"""
        # Start position: origin, 1m above ground
        self.position = np.array([0.0, 0.0, 1.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Orientation as Euler angles (roll, pitch, yaw)
        # Note: this still has gimbal lock issues, but good enough for now
        # Real implementation should use quaternions
        self.orientation = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # in body frame
        
        # Set or randomize target
        if self.fixed_target:
            self.target = self.fixed_target_position.copy()
        else:
            self.target = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(1, 5)
            ])
        
        if not self.fixed_obstacles:
            self._generate_random_obstacles()
        
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.position - self.target)
        self.trail.clear()
        self.hover_counter = 0
        self.reached_target = False
        
        # Reset LTL monitor
        if self.ltl_monitor:
            self.ltl_monitor.reset_all()
        
        return self._get_state()
    
    def _rotation_matrix(self, roll, pitch, yaw):
        """
        Compute proper 3D rotation matrix from Euler angles
        Uses ZYX convention (yaw-pitch-roll)
        
        This is the CORRECT way to transform body frame to world frame
        """
        # Rotation around Z (yaw)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        
        # Rotation around Y (pitch)
        R_y = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [ 0,             1, 0            ],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Rotation around X (roll)
        R_x = np.array([
            [1, 0,             0            ],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        
        # Combined rotation: R = R_z * R_y * R_x
        return R_z @ R_y @ R_x
    
    def step(self, action):
        """
        Simulate one physics step
        
        Args:
            action: array of 4 motor thrust commands [0, 1]
        
        Returns:
            next_state, reward, done, info
        """
        # Clamp and scale actions to actual thrust forces
        thrusts = np.clip(action, 0, 1) * self.max_thrust_per_motor
        self.motor_thrusts = thrusts.copy()
        
        # === Compute Torques ===
        # Motor layout (X-config):
        #     1(FL)  0(FR)
        #        \ /
        #        / \
        #     2(BL)  3(BR)
        
        # Roll torque: left motors - right motors
        roll_torque = self.arm_length * ((thrusts[1] + thrusts[2]) - (thrusts[0] + thrusts[3]))
        
        # Pitch torque: front motors - back motors
        pitch_torque = self.arm_length * ((thrusts[0] + thrusts[1]) - (thrusts[2] + thrusts[3]))
        
        # Yaw torque: from motor reaction torques (CW vs CCW)
        # Motors 0,2 spin CW, motors 1,3 spin CCW
        # Simplified model: yaw torque proportional to thrust difference
        yaw_torque = 0.05 * ((thrusts[0] + thrusts[2]) - (thrusts[1] + thrusts[3]))
        
        torques = np.array([roll_torque, pitch_torque, yaw_torque])
        
        # === Angular Dynamics ===
        # dω/dt = I^-1 * (τ - ω × (I*ω))  [Euler's rotation equation]
        # We're simplifying by ignoring the gyroscopic term (ω × (I*ω))
        # since it's small for slow rotations
        angular_acceleration = self.inertia_inv @ torques
        
        self.angular_velocity += angular_acceleration * self.dt
        self.angular_velocity *= (1 - self.angular_drag)  # damping
        
        # Update orientation (Euler integration)
        self.orientation += self.angular_velocity * self.dt
        
        # Wrap angles to [-π, π]
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))
        
        # === Linear Dynamics ===
        roll, pitch, yaw = self.orientation
        
        # Total thrust in body frame (pointing up in body coordinates)
        total_thrust = np.sum(thrusts)
        thrust_body = np.array([0, 0, total_thrust])
        
        # Transform thrust to world frame using CORRECT rotation matrix
        R = self._rotation_matrix(roll, pitch, yaw)
        thrust_world = R @ thrust_body
        
        # Gravity (always points down in world frame)
        gravity_force = np.array([0, 0, -self.mass * self.gravity])
        
        # Drag force (simplified linear model)
        # Real quadcopter has complex aerodynamics, but this is good enough
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            drag_direction = -self.velocity / velocity_magnitude
            drag_force = drag_direction * self.linear_drag_coeff * velocity_magnitude * velocity_magnitude
        else:
            drag_force = np.zeros(3)
        
        # Total force
        total_force = thrust_world + gravity_force + drag_force
        
        # F = ma -> a = F/m
        acceleration = total_force / self.mass
        
        # Update velocity and position (Euler integration)
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # Record trail for visualization
        self.trail.append(self.position.copy())
        self.current_step += 1
        
        # === Calculate Rewards ===
        base_reward, done = self._calculate_base_reward()
        
        # LTL reward shaping (if monitor exists)
        ltl_reward = 0.0
        violations = []
        satisfactions = []
        
        if self.ltl_monitor:
            ltl_reward, violations, satisfactions = self.ltl_monitor.evaluate_all(self)
            self.ltl_reward_history.append(ltl_reward)
            
            if violations:
                self.ltl_violations_history.extend(violations)
        
        total_reward = base_reward + ltl_reward
        
        info = {
            'base_reward': base_reward,
            'ltl_reward': ltl_reward,
            'violations': violations,
            'satisfactions': satisfactions
        }
        
        return self._get_state(), total_reward, done, info
    
    def _get_state(self):
        """
        Construct state observation vector
        
        State components:
        - Position (normalized)
        - Orientation (roll, pitch, yaw)
        - Velocity (normalized)
        - Angular velocity (normalized)
        - Relative position to target
        - Nearest 3 obstacles (position + distance)
        
        Total: 3 + 3 + 3 + 3 + 3 + 12 = 27 dimensions
        """
        # Normalize to roughly [-1, 1] range for better neural network training
        pos_norm = self.position / self.max_pos
        vel_norm = self.velocity / self.max_vel
        ori_norm = self.orientation / self.max_angle
        ang_vel_norm = self.angular_velocity / 10.0
        rel_pos = (self.target - self.position) / self.max_pos
        
        # Find 3 nearest obstacles
        obstacle_features = []
        if len(self.obstacles) > 0:
            distances = [(obs.distance_to(self.position), obs) for obs in self.obstacles]
            distances.sort(key=lambda x: x[0])
            
            for i in range(min(3, len(distances))):
                dist, obs = distances[i]
                rel_obstacle_pos = (obs.position - self.position) / self.max_pos
                obstacle_features.extend(rel_obstacle_pos)
                obstacle_features.append(dist / 5.0)  # normalized distance
        
        # Pad if fewer than 3 obstacles
        while len(obstacle_features) < 12:
            obstacle_features.extend([0.0, 0.0, 0.0, 1.0])  # far away placeholder
        
        state = np.concatenate([
            pos_norm,
            ori_norm,
            vel_norm,
            ang_vel_norm,
            rel_pos,
            obstacle_features
        ]).astype(np.float32)
        
        return state
    
    def _calculate_base_reward(self):
        """
        Base reward function (not including LTL shaping)
        
        Reward structure:
        - Progress toward target
        - Bonus for reaching target
        - Strong reward for stable hovering at target
        - Penalties for crashes, going out of bounds
        """
        distance = np.linalg.norm(self.position - self.target)
        velocity_magnitude = np.linalg.norm(self.velocity)
        tilt = np.abs(self.orientation[0]) + np.abs(self.orientation[1])
        angular_vel_magnitude = np.linalg.norm(self.angular_velocity)
        
        reward = 0.0
        done = False
        
        # Check collisions (also handled by LTL but need immediate done signal)
        for obs in self.obstacles:
            if obs.check_collision(self.position):
                reward -= 50
                done = True
                return reward, done
        
        # Hover logic
        in_position = distance < self.hover_zone_radius
        is_slow = velocity_magnitude < self.max_hover_velocity
        is_level = tilt < self.max_hover_tilt
        is_stable = angular_vel_magnitude < 0.5
        is_hovering_stably = in_position and is_slow and is_level and is_stable
        
        if in_position:
            self.reached_target = True
            reward += 15.0
            
            # Strong rewards for being still at target
            if is_slow:
                reward += 10.0
                reward -= velocity_magnitude * 20
            else:
                reward -= velocity_magnitude * 30
            
            # Strong reward for level orientation
            if is_level:
                reward += 10.0
            else:
                reward -= tilt * 40
            
            # Reward for stability
            if is_stable:
                reward += 8.0
            else:
                reward -= angular_vel_magnitude * 15
            
            # Big reward for perfect stable hover
            if is_hovering_stably:
                reward += 25.0
                self.hover_counter += 1
                
                # SUCCESS: hovered stably for required time
                if self.hover_counter >= self.hover_time_required:
                    reward += 500
                    done = True
                    return reward, done
            else:
                # Not stable, decay counter
                self.hover_counter = max(0, self.hover_counter - 2)
            
            # Penalty for drift from target center
            reward -= distance * 10
        
        else:
            # Not at target yet
            
            if self.reached_target:
                # Was at target but drifted away - bad!
                reward -= 100
                done = True
                return reward, done
            
            # Still approaching
            progress = (self.prev_distance - distance)
            reward += progress * 25
            
            # Distance penalty
            reward -= distance * 0.2
            
            # Bonus for getting close
            if distance < 1.0:
                reward += 8.0
            if distance < 2.0:
                reward += 3.0
        
        # Small alive bonus
        reward += 0.1
        
        # Penalty for excessive speed when not at target
        if not in_position and velocity_magnitude > 5.0:
            reward -= (velocity_magnitude - 5.0) * 0.2
        
        # === FAILURE CONDITIONS ===
        # Ground crash
        if self.position[2] < 0.05:
            reward -= 20
            done = True
        # Flew too high
        if self.position[2] > 20:
            reward -= 20
            done = True
        # Out of bounds horizontally
        if np.abs(self.position[0]) > 15 or np.abs(self.position[1]) > 15:
            reward -= 20
            done = True
        # Flipped over
        if np.abs(self.orientation[0]) > np.pi*0.8 or np.abs(self.orientation[1]) > np.pi*0.8:
            reward -= 15
            done = True
        # Timeout
        if self.current_step >= self.max_steps:
            done = True
        
        self.prev_distance = distance
        return reward, done
    
    def render(self):
        """Render the environment using pygame"""
        self.screen.fill((15, 15, 25))
        
        self._render_top_view(50, 50, 500, 500)
        self._render_side_view(600, 50, 500, 250)
        self._render_info_panel(600, 320)
        self._render_motor_indicators(600, 550)
        self._render_ltl_panel(1150, 50)
        
        pygame.display.flip()
        self.clock.tick(50)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True
    
    def _render_top_view(self, x_offset, y_offset, width, height):
        """Render top-down view (X-Y plane)"""
        pygame.draw.rect(self.screen, (25, 25, 35), (x_offset, y_offset, width, height))
        pygame.draw.rect(self.screen, (60, 60, 80), (x_offset, y_offset, width, height), 2)
        
        title = self.font.render("TOP VIEW (X-Y) - Fixed Physics", True, (200, 200, 220))
        self.screen.blit(title, (x_offset + 10, y_offset + 10))
        
        scale = 25
        center_x = x_offset + width // 2
        center_y = y_offset + height // 2
        
        # Grid
        for i in range(-10, 11, 2):
            screen_x = center_x + i * scale
            if x_offset < screen_x < x_offset + width:
                color = (40, 40, 50) if i != 0 else (60, 60, 80)
                pygame.draw.line(self.screen, color, (screen_x, y_offset), (screen_x, y_offset + height), 1)
            screen_y = center_y + i * scale
            if y_offset < screen_y < y_offset + height:
                color = (40, 40, 50) if i != 0 else (60, 60, 80)
                pygame.draw.line(self.screen, color, (x_offset, screen_y), (x_offset + width, screen_y), 1)
        
        # Obstacles
        for obs in self.obstacles:
            ox = int(center_x + obs.position[0] * scale)
            oy = int(center_y + obs.position[1] * scale)
            
            if obs.shape == 'sphere':
                radius = int(obs.size * scale)
                pygame.draw.circle(self.screen, (150, 50, 50), (ox, oy), radius)
                pygame.draw.circle(self.screen, (200, 80, 80), (ox, oy), radius, 2)
            elif obs.shape == 'box':
                w = int(obs.size[0] * scale)
                h = int(obs.size[1] * scale)
                rect = pygame.Rect(ox - w//2, oy - h//2, w, h)
                pygame.draw.rect(self.screen, (150, 50, 50), rect)
                pygame.draw.rect(self.screen, (200, 80, 80), rect, 2)
        
        # Trail
        if len(self.trail) > 1:
            points = [(int(center_x + pos[0] * scale), int(center_y + pos[1] * scale)) 
                     for pos in self.trail]
            if len(points) > 1:
                pygame.draw.lines(self.screen, (80, 120, 255), False, points, 2)
        
        # Target
        tx = int(center_x + self.target[0] * scale)
        ty = int(center_y + self.target[1] * scale)
        hover_radius = int(self.hover_zone_radius * scale)
        pygame.draw.circle(self.screen, (100, 255, 100), (tx, ty), hover_radius, 3)
        pygame.draw.circle(self.screen, (255, 80, 80), (tx, ty), 10)
        
        # Drone
        dx = int(center_x + self.position[0] * scale)
        dy = int(center_y + self.position[1] * scale)
        
        distance = np.linalg.norm(self.position - self.target)
        velocity_magnitude = np.linalg.norm(self.velocity)
        tilt = np.abs(self.orientation[0]) + np.abs(self.orientation[1])
        
        in_position = distance < self.hover_zone_radius
        is_slow = velocity_magnitude < self.max_hover_velocity
        is_level = tilt < self.max_hover_tilt
        
        # Color based on status
        if in_position and is_slow and is_level:
            color = (100, 255, 100)
        elif in_position:
            color = (255, 255, 100)
        else:
            color = (100, 150, 255)
        
        # Size based on altitude
        height_ratio = np.clip(self.position[2] / 5.0, 0, 1)
        size = int(8 + 8 * height_ratio)
        
        # Draw arms (X configuration)
        yaw = self.orientation[2]
        arm_len = size + 12
        
        # 4 arms at 45, 135, 225, 315 degrees (plus yaw rotation)
        angles = [yaw + np.pi/4, yaw + 3*np.pi/4, yaw + 5*np.pi/4, yaw + 7*np.pi/4]
        for angle in angles:
            arm_x = int(dx + arm_len * np.cos(angle))
            arm_y = int(dy + arm_len * np.sin(angle))
            pygame.draw.line(self.screen, color, (dx, dy), (arm_x, arm_y), 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (arm_x, arm_y), 4)
        
        # Center body
        pygame.draw.circle(self.screen, color, (dx, dy), size)
        pygame.draw.circle(self.screen, (255, 255, 255), (dx, dy), size + 2, 1)
        
        # Front indicator (yellow nose pointing forward)
        nose_len = size + 8
        nose_x = int(dx + nose_len * np.cos(yaw))
        nose_y = int(dy + nose_len * np.sin(yaw))
        pygame.draw.circle(self.screen, (255, 255, 100), (nose_x, nose_y), 5)
    
    def _render_side_view(self, x_offset, y_offset, width, height):
        """Render side view (X-Z plane)"""
        pygame.draw.rect(self.screen, (25, 25, 35), (x_offset, y_offset, width, height))
        pygame.draw.rect(self.screen, (60, 60, 80), (x_offset, y_offset, width, height), 2)
        
        title = self.font.render("SIDE VIEW (X-Z)", True, (200, 200, 220))
        self.screen.blit(title, (x_offset + 10, y_offset + 10))
        
        scale = 20
        center_x = x_offset + width // 2
        ground_y = y_offset + height - 30
        
        # Ground line
        pygame.draw.line(self.screen, (100, 100, 80), (x_offset, ground_y), (x_offset + width, ground_y), 3)
        
        # Obstacles
        for obs in self.obstacles:
            ox = int(center_x + obs.position[0] * scale)
            oz = int(ground_y - obs.position[2] * scale)
            
            if obs.shape == 'sphere':
                radius = int(obs.size * scale)
                pygame.draw.circle(self.screen, (150, 50, 50), (ox, oz), radius)
            elif obs.shape == 'box':
                w = int(obs.size[0] * scale)
                h = int(obs.size[2] * scale)
                rect = pygame.Rect(ox - w//2, oz - h//2, w, h)
                pygame.draw.rect(self.screen, (150, 50, 50), rect)
        
        # Target
        tx = int(center_x + self.target[0] * scale)
        tz = int(ground_y - self.target[2] * scale)
        pygame.draw.circle(self.screen, (255, 80, 80), (tx, tz), 10)
        
        # Drone
        dx = int(center_x + self.position[0] * scale)
        dz = int(ground_y - self.position[2] * scale)
        
        distance = np.linalg.norm(self.position - self.target)
        velocity_magnitude = np.linalg.norm(self.velocity)
        tilt = np.abs(self.orientation[0]) + np.abs(self.orientation[1])
        
        in_position = distance < self.hover_zone_radius
        is_slow = velocity_magnitude < self.max_hover_velocity
        is_level = tilt < self.max_hover_tilt
        
        if in_position and is_slow and is_level:
            drone_color = (100, 255, 100)
        elif in_position:
            drone_color = (255, 255, 100)
        else:
            drone_color = (100, 150, 255)
        
        # Draw drone with tilt visualization (roll angle)
        roll = self.orientation[0]
        tilt_len = 20
        tx1 = int(dx - tilt_len * np.cos(roll))
        tz1 = int(dz + tilt_len * np.sin(roll))
        tx2 = int(dx + tilt_len * np.cos(roll))
        tz2 = int(dz - tilt_len * np.sin(roll))
        
        pygame.draw.line(self.screen, drone_color, (tx1, tz1), (tx2, tz2), 6)
        pygame.draw.circle(self.screen, drone_color, (dx, dz), 10)
        pygame.draw.circle(self.screen, (255, 255, 255), (dx, dz), 12, 2)
    
    def _render_info_panel(self, x_offset, y_offset):
        """Render status information panel"""
        distance = np.linalg.norm(self.position - self.target)
        velocity_magnitude = np.linalg.norm(self.velocity)
        tilt = np.abs(self.orientation[0]) + np.abs(self.orientation[1])
        hover_progress = (self.hover_counter / self.hover_time_required) * 100
        
        in_position = distance < self.hover_zone_radius
        is_slow = velocity_magnitude < self.max_hover_velocity
        is_level = tilt < self.max_hover_tilt
        
        status = "🔵 Approaching"
        if in_position:
            if is_slow and is_level:
                status = "🟢 STABLE HOVER"
            else:
                status = "🟡 At Target"
        
        avg_ltl_reward = np.mean(self.ltl_reward_history) if self.ltl_reward_history else 0
        
        info_lines = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Status: {status}",
            f"Hover: {hover_progress:.0f}% ({self.hover_counter}/{self.hover_time_required})",
            f"",
            f"Distance: {distance:.3f}m",
            f"Velocity: {velocity_magnitude:.3f} m/s",
            f"Tilt: {np.degrees(tilt):.1f}°",
            f"",
            f"LTL Avg Reward: {avg_ltl_reward:.1f}",
            f"Violations: {len(self.ltl_violations_history)}",
        ]
        
        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, (200, 200, 220))
            self.screen.blit(text, (x_offset, y_offset + i * 22))
    
    def _render_motor_indicators(self, x_offset, y_offset):
        """Render motor thrust indicators"""
        title = self.font.render("MOTOR THRUSTS", True, (200, 200, 220))
        self.screen.blit(title, (x_offset, y_offset))
        
        motor_names = ["FR", "FL", "BL", "BR"]
        bar_width = 150
        bar_height = 18
        
        for i in range(4):
            y = y_offset + 30 + i * 25
            pygame.draw.rect(self.screen, (40, 40, 50), (x_offset, y, bar_width, bar_height))
            
            thrust_ratio = self.motor_thrusts[i] / self.max_thrust_per_motor
            thrust_width = int(bar_width * thrust_ratio)
            color = (int(50 + 200 * thrust_ratio), int(100 + 150 * thrust_ratio), int(255 - 100 * thrust_ratio))
            
            if thrust_width > 0:
                pygame.draw.rect(self.screen, color, (x_offset, y, thrust_width, bar_height))
            
            pygame.draw.rect(self.screen, (100, 100, 120), (x_offset, y, bar_width, bar_height), 1)
            label = self.small_font.render(f"{motor_names[i]}: {self.motor_thrusts[i]:.2f}N", True, (200, 200, 220))
            self.screen.blit(label, (x_offset + bar_width + 10, y + 2))
    
    def _render_ltl_panel(self, x_offset, y_offset):
        """Render LTL specification status"""
        if not self.ltl_monitor:
            return
        
        pygame.draw.rect(self.screen, (30, 30, 40), (x_offset, y_offset, 230, 700))
        pygame.draw.rect(self.screen, (80, 80, 100), (x_offset, y_offset, 230, 700), 2)
        
        title = self.font.render("LTL Specs", True, (200, 200, 255))
        self.screen.blit(title, (x_offset + 10, y_offset + 10))
        
        y = y_offset + 45
        
        for spec in self.ltl_monitor.specifications:
            # Spec name
            name_text = self.small_font.render(spec.name[:18], True, (220, 220, 240))
            self.screen.blit(name_text, (x_offset + 10, y))
            y += 20
            
            # Type
            type_color = {
                'safety': (255, 100, 100),
                'liveness': (100, 255, 100),
                'response': (255, 255, 100),
                'persistence': (100, 200, 255)
            }
            color = type_color.get(spec.property_type.value, (150, 150, 150))
            type_text = self.tiny_font.render(spec.property_type.value.upper(), True, color)
            self.screen.blit(type_text, (x_offset + 15, y))
            y += 18
            
            # Stats
            total = spec.violations + spec.satisfactions
            if total > 0:
                compliance = spec.satisfactions / total * 100
                stats_text = self.tiny_font.render(
                    f"✓{spec.satisfactions} ✗{spec.violations} ({compliance:.0f}%)", 
                    True, (180, 180, 200)
                )
                self.screen.blit(stats_text, (x_offset + 15, y))
            
            y += 25
            
            if y > y_offset + 680:
                break
    
    def close(self):
        """Clean up pygame"""
        pygame.quit()