"""
Realistic drone physics with FIXED obstacles AND FIXED target!
Drone must REACH target and HOVER STABLY (stay in place, minimal drift).
"""
import numpy as np
import pygame
from collections import deque

class Obstacle:
    """Represents a 3D obstacle (box or sphere)"""
    def __init__(self, position, size, shape='box'):
        self.position = np.array(position)  # [x, y, z]
        self.size = size  # For box: [width, depth, height], for sphere: radius
        self.shape = shape  # 'box' or 'sphere'
    
    def check_collision(self, point, safety_margin=0.3):
        """Check if a point collides with this obstacle"""
        if self.shape == 'sphere':
            distance = np.linalg.norm(point - self.position)
            return distance < (self.size + safety_margin)
        
        elif self.shape == 'box':
            # Check if point is inside box (with margin)
            half_size = np.array(self.size) / 2 + safety_margin
            diff = np.abs(point - self.position)
            return np.all(diff < half_size)
        
        return False
    
    def distance_to(self, point):
        """Calculate minimum distance from point to obstacle surface"""
        if self.shape == 'sphere':
            return max(0, np.linalg.norm(point - self.position) - self.size)
        
        elif self.shape == 'box':
            # Distance to box surface
            half_size = np.array(self.size) / 2
            diff = np.abs(point - self.position) - half_size
            outside_distance = np.maximum(diff, 0)
            return np.linalg.norm(outside_distance)

class RealisticDroneEnv:
    def __init__(self, num_obstacles=5, fixed_obstacles=True, fixed_target=True):
        # Physical constants
        self.mass = 1.0
        self.gravity = 9.81
        self.dt = 0.02
        
        # Drag coefficients
        self.linear_drag = 0.1
        self.angular_drag = 0.05
        
        # Motor parameters
        self.arm_length = 0.25
        self.max_thrust_per_motor = 4.0
        self.inertia = 0.01
        
        # Simulation bounds
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
        
        # Target configuration
        self.fixed_target = fixed_target
        self.fixed_target_position = np.array([7.0, 3.0, 3.0])  # FIXED TARGET POSITION
        
        # Hovering behavior - must stay STABLE at target
        self.hover_zone_radius = 0.3  # Must stay within 30cm sphere
        self.max_hover_velocity = 0.3  # Max 30cm/s velocity when hovering
        self.max_hover_tilt = 0.15  # Max ~8.6 degrees tilt
        self.hover_time_required = 100  # Must hover for 100 steps (2 seconds)
        self.hover_counter = 0
        self.reached_target = False
        
        # Generate fixed obstacles once
        if self.fixed_obstacles:
            self._generate_fixed_obstacles()
        
        # Rendering
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 700
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("🚁 Drone RL - Stable Hover")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Visualization state
        self.trail = deque(maxlen=50)
        self.motor_thrusts = np.zeros(4)
        
        self.reset()
    
    def _generate_fixed_obstacles(self):
        """Generate FIXED obstacles that stay the same every episode"""
        self.obstacles = [
            # Side barriers
            
            # Floating spheres
            Obstacle([-6, 3.0, 2.0], 0.7, 'sphere'),
        ]
        
        # Use only the requested number
        self.obstacles = self.obstacles[:self.num_obstacles]
    
    def _generate_random_obstacles(self):
        """Generate random obstacles (if not using fixed)"""
        self.obstacles = []
        
        for _ in range(self.num_obstacles):
            attempts = 0
            while attempts < 50:
                if np.random.rand() < 0.5:
                    # Box obstacle
                    pos = np.array([
                        np.random.uniform(-6, 6),
                        np.random.uniform(-6, 6),
                        np.random.uniform(0.5, 4)
                    ])
                    size = [
                        np.random.uniform(0.5, 2.0),
                        np.random.uniform(0.5, 2.0),
                        np.random.uniform(1.0, 3.0)
                    ]
                    obstacle = Obstacle(pos, size, 'box')
                else:
                    # Sphere obstacle
                    pos = np.array([
                        np.random.uniform(-6, 6),
                        np.random.uniform(-6, 6),
                        np.random.uniform(1.5, 4)
                    ])
                    radius = np.random.uniform(0.3, 0.8)
                    obstacle = Obstacle(pos, radius, 'sphere')
                
                # Check if obstacle is too close to spawn or target
                spawn_dist = np.linalg.norm(obstacle.position - np.array([0, 0, 1]))
                target_dist = np.linalg.norm(obstacle.position - self.target)
                
                if spawn_dist > 2.0 and target_dist > 1.5:
                    self.obstacles.append(obstacle)
                    break
                
                attempts += 1
    
    def reset(self):
        """Reset environment to initial state"""
        self.position = np.array([0.0, 0.0, 1.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        
        # Use fixed or random target
        if self.fixed_target:
            self.target = self.fixed_target_position.copy()
        else:
            self.target = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(1, 5)
            ])
        
        # Only regenerate obstacles if not using fixed ones
        if not self.fixed_obstacles:
            self._generate_random_obstacles()
        
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.position - self.target)
        self.trail.clear()
        
        # Reset hovering state
        self.hover_counter = 0
        self.reached_target = False
        
        return self._get_state()
    
    def step(self, action):
        """Execute one timestep with realistic physics"""
        # Clip and scale actions
        thrusts = np.clip(action, 0, 1) * self.max_thrust_per_motor
        self.motor_thrusts = thrusts.copy()
        
        # ===== PHYSICS SIMULATION =====
        total_thrust = np.sum(thrusts)
        
        # Torques (X configuration motors)
        roll_torque = self.arm_length * ((thrusts[1] + thrusts[2]) - (thrusts[0] + thrusts[3]))
        pitch_torque = self.arm_length * ((thrusts[0] + thrusts[1]) - (thrusts[2] + thrusts[3]))
        yaw_torque = 0.05 * ((thrusts[0] + thrusts[2]) - (thrusts[1] + thrusts[3]))
        
        # Update angular velocities
        self.angular_velocity += np.array([roll_torque, pitch_torque, yaw_torque]) / self.inertia * self.dt
        self.angular_velocity *= (1 - self.angular_drag)
        
        # Update orientation
        self.orientation += self.angular_velocity * self.dt
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))
        
        # Thrust in world frame
        roll, pitch, yaw = self.orientation
        thrust_world = np.array([
            total_thrust * np.sin(roll) * np.cos(yaw),
            total_thrust * np.sin(pitch) * np.sin(yaw),
            total_thrust * np.cos(roll) * np.cos(pitch)
        ])
        
        # Forces
        gravity_force = np.array([0, 0, -self.mass * self.gravity])
        drag_force = -self.linear_drag * self.velocity * np.abs(self.velocity)
        total_force = thrust_world + gravity_force + drag_force
        
        # Update velocity and position
        acceleration = total_force / self.mass
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # Add to trail
        self.trail.append(self.position.copy())
        
        self.current_step += 1
        
        # Calculate reward
        reward, done = self._calculate_reward()
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        """Get normalized observation including obstacle information"""
        pos_norm = self.position / self.max_pos
        vel_norm = self.velocity / self.max_vel
        ori_norm = self.orientation / self.max_angle
        ang_vel_norm = self.angular_velocity / 10.0
        rel_pos = (self.target - self.position) / self.max_pos
        
        # Find closest obstacles (top 3)
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
        while len(obstacle_features) < 12:  # 3 obstacles * 4 features each
            obstacle_features.extend([0.0, 0.0, 0.0, 1.0])  # far away placeholder
        
        state = np.concatenate([
            pos_norm,           # 3
            ori_norm,           # 3
            vel_norm,           # 3
            ang_vel_norm,       # 3
            rel_pos,            # 3
            obstacle_features   # 12 (3 closest obstacles)
        ]).astype(np.float32)
        
        return state  # Total: 27 dimensions
    
    def _calculate_reward(self):
        """Reward function that encourages reaching target and STAYING STABLE"""
        distance = np.linalg.norm(self.position - self.target)
        velocity_magnitude = np.linalg.norm(self.velocity)
        tilt = np.abs(self.orientation[0]) + np.abs(self.orientation[1])
        angular_vel_magnitude = np.linalg.norm(self.angular_velocity)
        
        reward = 0.0
        done = False
        
        # ===== COLLISION CHECK =====
        for obs in self.obstacles:
            if obs.check_collision(self.position):
                reward -= 100
                done = True
                print("💥 COLLISION with obstacle!")
                return reward, done
            
            # Proximity penalty
            dist_to_obs = obs.distance_to(self.position)
            if dist_to_obs < 0.3:
                reward -= (0.3 - dist_to_obs) * 2
        
        # ===== CHECK IF IN STABLE HOVER =====
        in_position = distance < self.hover_zone_radius
        is_slow = velocity_magnitude < self.max_hover_velocity
        is_level = tilt < self.max_hover_tilt
        is_stable = angular_vel_magnitude < 0.5
        
        is_hovering_stably = in_position and is_slow and is_level and is_stable
        
        if in_position:
            self.reached_target = True
            
            # REWARD: In the target zone
            reward += 15.0
            
            # REWARD: Low velocity (staying in place)
            if is_slow:
                reward += 10.0
                reward -= velocity_magnitude * 10  # Penalize any movement
            else:
                # Too fast in target zone!
                reward -= velocity_magnitude * 20
            
            # REWARD: Level orientation
            if is_level:
                reward += 8.0
            else:
                reward -= tilt * 30  # Heavy penalty for tilting
            
            # REWARD: Stable (not spinning)
            if is_stable:
                reward += 5.0
            else:
                reward -= angular_vel_magnitude * 10
            
            # Check if PERFECTLY stable
            if is_hovering_stably:
                reward += 20.0  # BIG reward for perfect hover
                self.hover_counter += 1
                
                # SUCCESS: Hovered stably for required time
                if self.hover_counter >= self.hover_time_required:
                    reward += 1000  # MASSIVE SUCCESS
                    done = True
                    print(f"🎯✨ STABLE HOVER ACHIEVED for {self.hover_counter} steps!")
                    return reward, done
            else:
                # Not stable enough, decay counter
                self.hover_counter = max(0, self.hover_counter - 2)
        
        else:
            # NOT at target yet
            
            if self.reached_target:
                # HUGE PENALTY: Was at target but drifted away!
                reward -= 100
                print("❌ DRIFTED away from target!")
                done = True
                return reward, done
            
            # Still approaching
            progress = (self.prev_distance - distance)
            reward += progress * 30
            
            # Distance penalty
            reward -= distance * 0.1
            
            # Bonus for getting close
            if distance < 1.0:
                reward += 5.0
            if distance < 2.0:
                reward += 2.0
        
        # ===== GENERAL PENALTIES =====
        # Small alive bonus
        reward += 0.1
        
        # Penalty for excessive speed when not at target
        if not in_position and velocity_magnitude > 5.0:
            reward -= (velocity_magnitude - 5.0) * 0.1
        
        # ===== FAILURE CONDITIONS =====
        if self.position[2] < 0.05:
            reward -= 50
            done = True
            print("💥 Hit ground!")
        if self.position[2] > 20:
            reward -= 50
            done = True
        if np.abs(self.position[0]) > 15 or np.abs(self.position[1]) > 15:
            reward -= 50
            done = True
        if np.abs(self.orientation[0]) > np.pi*0.8 or np.abs(self.orientation[1]) > np.pi*0.8:
            reward -= 30
            done = True
            print("💥 Flipped over!")
        if self.current_step >= self.max_steps:
            done = True
        
        self.prev_distance = distance
        
        return reward, done
    
    def render(self):
        """Enhanced visualization with obstacles"""
        self.screen.fill((15, 15, 25))
        
        # ===== TOP VIEW (Left half) =====
        self._render_top_view(50, 50, 500, 500)
        
        # ===== SIDE VIEW (Right top) =====
        self._render_side_view(650, 50, 500, 250)
        
        # ===== INFO PANEL (Right bottom) =====
        self._render_info_panel(650, 350)
        
        # ===== MOTOR INDICATORS =====
        self._render_motor_indicators(650, 550)
        
        pygame.display.flip()
        self.clock.tick(50)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True
    
    def _render_top_view(self, x_offset, y_offset, width, height):
        """Render top-down view with obstacles"""
        pygame.draw.rect(self.screen, (25, 25, 35), (x_offset, y_offset, width, height))
        pygame.draw.rect(self.screen, (60, 60, 80), (x_offset, y_offset, width, height), 2)
        
        title_text = "TOP VIEW - Stable Hover Mode"
        title = self.font.render(title_text, True, (200, 200, 220))
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
        
        # ===== RENDER OBSTACLES =====
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
            points = []
            for pos in self.trail:
                sx = int(center_x + pos[0] * scale)
                sy = int(center_y + pos[1] * scale)
                points.append((sx, sy))
            if len(points) > 1:
                pygame.draw.lines(self.screen, (80, 120, 255, 100), False, points, 2)
        
        # Target with stability zone
        tx = int(center_x + self.target[0] * scale)
        ty = int(center_y + self.target[1] * scale)
        
        # Hover zone circle (must stay within this!)
        hover_radius = int(self.hover_zone_radius * scale)
        pygame.draw.circle(self.screen, (100, 255, 100), (tx, ty), hover_radius, 3)
        
        # Target center
        pygame.draw.circle(self.screen, (255, 80, 80), (tx, ty), 10)
        pygame.draw.circle(self.screen, (255, 120, 120), (tx, ty), 14, 2)
        
        # Drone
        dx = int(center_x + self.position[0] * scale)
        dy = int(center_y + self.position[1] * scale)
        
        distance = np.linalg.norm(self.position - self.target)
        velocity_magnitude = np.linalg.norm(self.velocity)
        tilt = np.abs(self.orientation[0]) + np.abs(self.orientation[1])
        
        # Color based on hover stability
        in_position = distance < self.hover_zone_radius
        is_slow = velocity_magnitude < self.max_hover_velocity
        is_level = tilt < self.max_hover_tilt
        
        if in_position and is_slow and is_level:
            color = (100, 255, 100)  # GREEN = stable hover
        elif in_position:
            color = (255, 255, 100)  # YELLOW = at target but not stable
        else:
            color = (100, 150, 255)  # BLUE = approaching
        
        height_ratio = np.clip(self.position[2] / 5.0, 0, 1)
        size = int(8 + 8 * height_ratio)
        
        pygame.draw.circle(self.screen, color, (dx, dy), size)
        pygame.draw.circle(self.screen, (255, 255, 255), (dx, dy), size + 2, 1)
        
        # Orientation arrow
        yaw = self.orientation[2]
        arrow_len = size + 15
        ax = int(dx + arrow_len * np.cos(yaw))
        ay = int(dy + arrow_len * np.sin(yaw))
        pygame.draw.line(self.screen, (255, 255, 100), (dx, dy), (ax, ay), 3)
        
        # Connection line to target
        pygame.draw.line(self.screen, (100, 100, 120), (dx, dy), (tx, ty), 1)
    
    def _render_side_view(self, x_offset, y_offset, width, height):
        """Render side view with obstacles"""
        pygame.draw.rect(self.screen, (25, 25, 35), (x_offset, y_offset, width, height))
        pygame.draw.rect(self.screen, (60, 60, 80), (x_offset, y_offset, width, height), 2)
        
        title = self.font.render("SIDE VIEW (X-Z)", True, (200, 200, 220))
        self.screen.blit(title, (x_offset + 10, y_offset + 10))
        
        scale = 20
        center_x = x_offset + width // 2
        ground_y = y_offset + height - 30
        
        # Ground line
        pygame.draw.line(self.screen, (100, 100, 80), (x_offset, ground_y), (x_offset + width, ground_y), 3)
        
        # Grid
        for i in range(0, 11):
            gy = int(ground_y - i * scale)
            if y_offset < gy < y_offset + height:
                pygame.draw.line(self.screen, (40, 40, 50), (x_offset, gy), (x_offset + width, gy), 1)
        
        # ===== RENDER OBSTACLES (side view) =====
        for obs in self.obstacles:
            ox = int(center_x + obs.position[0] * scale)
            oz = int(ground_y - obs.position[2] * scale)
            
            if obs.shape == 'sphere':
                radius = int(obs.size * scale)
                pygame.draw.circle(self.screen, (150, 50, 50), (ox, oz), radius)
                pygame.draw.circle(self.screen, (200, 80, 80), (ox, oz), radius, 1)
            
            elif obs.shape == 'box':
                w = int(obs.size[0] * scale)
                h = int(obs.size[2] * scale)  # height in z
                rect = pygame.Rect(ox - w//2, oz - h//2, w, h)
                pygame.draw.rect(self.screen, (150, 50, 50), rect)
                pygame.draw.rect(self.screen, (200, 80, 80), rect, 1)
        
        # Target
        tx = int(center_x + self.target[0] * scale)
        tz = int(ground_y - self.target[2] * scale)
        pygame.draw.circle(self.screen, (255, 80, 80), (tx, tz), 10)
        pygame.draw.line(self.screen, (255, 80, 80), (tx, tz), (tx, ground_y), 1)
        
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
            drone_color = (150, 200, 255)
        
        roll = self.orientation[0]
        tilt_len = 20
        tx1 = int(dx - tilt_len * np.cos(roll))
        tz1 = int(dz + tilt_len * np.sin(roll))
        tx2 = int(dx + tilt_len * np.cos(roll))
        tz2 = int(dz - tilt_len * np.sin(roll))
        
        pygame.draw.line(self.screen, drone_color, (tx1, tz1), (tx2, tz2), 6)
        pygame.draw.circle(self.screen, drone_color, (dx, dz), 10)
        
        pygame.draw.line(self.screen, (80, 80, 100), (dx, dz), (dx, ground_y), 1)
    
    def _render_info_panel(self, x_offset, y_offset):
        """Render information panel"""
        # Find closest obstacle
        closest_obs_dist = float('inf')
        if self.obstacles:
            closest_obs_dist = min(obs.distance_to(self.position) for obs in self.obstacles)
        
        distance = np.linalg.norm(self.position - self.target)
        velocity_magnitude = np.linalg.norm(self.velocity)
        tilt = np.abs(self.orientation[0]) + np.abs(self.orientation[1])
        hover_progress = (self.hover_counter / self.hover_time_required) * 100
        
        # Stability checks
        in_position = distance < self.hover_zone_radius
        is_slow = velocity_magnitude < self.max_hover_velocity
        is_level = tilt < self.max_hover_tilt
        
        status = "🔵 Approaching"
        if in_position:
            if is_slow and is_level:
                status = "🟢 STABLE HOVER"
            else:
                status = "🟡 At Target (Unstable)"
        
        info_lines = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Status: {status}",
            f"Hover Progress: {hover_progress:.0f}% ({self.hover_counter}/{self.hover_time_required})",
            f"",
            f"Distance to Target: {distance:.3f}m (max: {self.hover_zone_radius}m)",
            f"Velocity: {velocity_magnitude:.3f} m/s (max: {self.max_hover_velocity} m/s)",
            f"Tilt: {np.degrees(tilt):.1f}° (max: {np.degrees(self.max_hover_tilt):.1f}°)",
            f"",
            f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})",
            f"Target:   ({self.target[0]:.2f}, {self.target[1]:.2f}, {self.target[2]:.2f})",
            f"Closest Obstacle: {closest_obs_dist:.2f}m",
        ]
        
        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, (200, 200, 220))
            self.screen.blit(text, (x_offset, y_offset + i * 22))
  
    def _render_motor_indicators(self, x_offset, y_offset):
        """Render motor thrust bars"""
        title = self.font.render("MOTOR THRUSTS", True, (200, 200, 220))
        self.screen.blit(title, (x_offset, y_offset))
        
        motor_names = ["Front-Right", "Front-Left", "Back-Left", "Back-Right"]
        bar_width = 200
        bar_height = 20
        
        for i in range(4):
            y = y_offset + 35 + i * 30
            
            pygame.draw.rect(self.screen, (40, 40, 50), (x_offset, y, bar_width, bar_height))
            
            thrust_ratio = self.motor_thrusts[i] / self.max_thrust_per_motor
            thrust_width = int(bar_width * thrust_ratio)
            
            color = (
                int(50 + 200 * thrust_ratio),
                int(100 + 150 * thrust_ratio),
                int(255 - 100 * thrust_ratio)
            )
            
            if thrust_width > 0:
                pygame.draw.rect(self.screen, color, (x_offset, y, thrust_width, bar_height))
            
            pygame.draw.rect(self.screen, (100, 100, 120), (x_offset, y, bar_width, bar_height), 1)
            
            label = self.small_font.render(f"{motor_names[i]}: {self.motor_thrusts[i]:.2f}N", True, (200, 200, 220))
            self.screen.blit(label, (x_offset + bar_width + 10, y + 3))
    
    def close(self):
        """Clean up"""
        pygame.quit()