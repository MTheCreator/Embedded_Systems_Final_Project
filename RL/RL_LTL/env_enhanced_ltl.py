"""
Realistic drone physics with LTL specifications integrated
"""
import numpy as np
import pygame
from collections import deque

class Obstacle:
    """Represents a 3D obstacle (box or sphere)"""
    def __init__(self, position, size, shape='box'):
        self.position = np.array(position)
        self.size = size
        self.shape = shape
    
    def check_collision(self, point, safety_margin=0.3):
        if self.shape == 'sphere':
            distance = np.linalg.norm(point - self.position)
            return distance < (self.size + safety_margin)
        elif self.shape == 'box':
            half_size = np.array(self.size) / 2 + safety_margin
            diff = np.abs(point - self.position)
            return np.all(diff < half_size)
        return False
    
    def distance_to(self, point):
        if self.shape == 'sphere':
            return max(0, np.linalg.norm(point - self.position) - self.size)
        elif self.shape == 'box':
            half_size = np.array(self.size) / 2
            diff = np.abs(point - self.position) - half_size
            outside_distance = np.maximum(diff, 0)
            return np.linalg.norm(outside_distance)

class RealisticDroneEnvLTL:
    def __init__(self, num_obstacles=5, fixed_obstacles=True, fixed_target=True, ltl_monitor=None):
        # Physical constants
        self.mass = 1.0
        self.gravity = 9.81
        self.dt = 0.02
        
        self.linear_drag = 0.1
        self.angular_drag = 0.05
        
        self.arm_length = 0.25
        self.max_thrust_per_motor = 4.0
        self.inertia = 0.01
        
        self.max_pos = 10.0
        self.max_vel = 20.0
        self.max_angle = np.pi
        
        self.max_steps = 2000
        self.current_step = 0
        
        # Obstacles
        self.num_obstacles = num_obstacles
        self.fixed_obstacles = fixed_obstacles
        self.obstacles = []
        
        # Target
        self.fixed_target = fixed_target
        self.fixed_target_position = np.array([7.0, 3.0, 3.0])
        
        # Hovering
        self.hover_zone_radius = 0.3
        self.max_hover_velocity = 0.3
        self.max_hover_tilt = 0.15
        self.hover_time_required = 100
        self.hover_counter = 0
        self.reached_target = False
        
        # LTL Monitor
        self.ltl_monitor = ltl_monitor
        self.ltl_reward_history = deque(maxlen=100)
        self.ltl_violations_history = []
        
        # Generate obstacles
        if self.fixed_obstacles:
            self._generate_fixed_obstacles()
        
        # Rendering
        pygame.init()
        self.screen_width = 1400
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("🚁 Drone RL with LTL Specifications")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 16)
        
        self.trail = deque(maxlen=50)
        self.motor_thrusts = np.zeros(4)
        
        self.reset()
    
    def _generate_fixed_obstacles(self):
        self.obstacles = [
            Obstacle([-6, 3.0, 2.0], 0.7, 'sphere'),
            Obstacle([2, 0, 2.5], [1.5, 1.5, 3.0], 'box'),
            
        ]
        self.obstacles = self.obstacles[:self.num_obstacles]
    
    def reset(self):
        self.position = np.array([0.0, 0.0, 1.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        
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
    
    def step(self, action):
        thrusts = np.clip(action, 0, 1) * self.max_thrust_per_motor
        self.motor_thrusts = thrusts.copy()
        
        # Physics
        total_thrust = np.sum(thrusts)
        
        roll_torque = self.arm_length * ((thrusts[1] + thrusts[2]) - (thrusts[0] + thrusts[3]))
        pitch_torque = self.arm_length * ((thrusts[0] + thrusts[1]) - (thrusts[2] + thrusts[3]))
        yaw_torque = 0.05 * ((thrusts[0] + thrusts[2]) - (thrusts[1] + thrusts[3]))
        
        self.angular_velocity += np.array([roll_torque, pitch_torque, yaw_torque]) / self.inertia * self.dt
        self.angular_velocity *= (1 - self.angular_drag)
        
        self.orientation += self.angular_velocity * self.dt
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))
        
        roll, pitch, yaw = self.orientation
        thrust_world = np.array([
            total_thrust * np.sin(roll) * np.cos(yaw),
            total_thrust * np.sin(pitch) * np.sin(yaw),
            total_thrust * np.cos(roll) * np.cos(pitch)
        ])
        
        gravity_force = np.array([0, 0, -self.mass * self.gravity])
        drag_force = -self.linear_drag * self.velocity * np.abs(self.velocity)
        total_force = thrust_world + gravity_force + drag_force
        
        acceleration = total_force / self.mass
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        
        self.trail.append(self.position.copy())
        self.current_step += 1
        
        # Calculate base reward
        base_reward, done = self._calculate_base_reward()
        
        # Add LTL reward shaping
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
        pos_norm = self.position / self.max_pos
        vel_norm = self.velocity / self.max_vel
        ori_norm = self.orientation / self.max_angle
        ang_vel_norm = self.angular_velocity / 10.0
        rel_pos = (self.target - self.position) / self.max_pos
        
        obstacle_features = []
        if len(self.obstacles) > 0:
            distances = [(obs.distance_to(self.position), obs) for obs in self.obstacles]
            distances.sort(key=lambda x: x[0])
            
            for i in range(min(3, len(distances))):
                dist, obs = distances[i]
                rel_obstacle_pos = (obs.position - self.position) / self.max_pos
                obstacle_features.extend(rel_obstacle_pos)
                obstacle_features.append(dist / 5.0)
        
        while len(obstacle_features) < 12:
            obstacle_features.extend([0.0, 0.0, 0.0, 1.0])
        
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
        """Base reward function (original logic)"""
        distance = np.linalg.norm(self.position - self.target)
        velocity_magnitude = np.linalg.norm(self.velocity)
        tilt = np.abs(self.orientation[0]) + np.abs(self.orientation[1])
        angular_vel_magnitude = np.linalg.norm(self.angular_velocity)
        
        reward = 0.0
        done = False
        
        # Check collisions (handled by LTL, but keep for immediate done)
        for obs in self.obstacles:
            if obs.check_collision(self.position):
                reward -= 50  # Reduced since LTL adds penalty
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
            reward += 10.0
            
            if is_slow:
                reward += 5.0
                reward -= velocity_magnitude * 5
            else:
                reward -= velocity_magnitude * 10
            
            if is_level:
                reward += 5.0
            else:
                reward -= tilt * 15
            
            if is_stable:
                reward += 3.0
            else:
                reward -= angular_vel_magnitude * 5
            
            if is_hovering_stably:
                reward += 15.0
                self.hover_counter += 1
                
                if self.hover_counter >= self.hover_time_required:
                    reward += 500  # Reduced since LTL adds reward
                    done = True
                    return reward, done
            else:
                self.hover_counter = max(0, self.hover_counter - 2)
        else:
            if self.reached_target:
                reward -= 50  # Reduced since LTL adds penalty
                done = True
                return reward, done
            
            progress = (self.prev_distance - distance)
            reward += progress * 20
            reward -= distance * 0.05
            
            if distance < 1.0:
                reward += 3.0
            if distance < 2.0:
                reward += 1.0
        
        reward += 0.1
        
        # Failure conditions (handled by LTL but keep for immediate termination)
        if self.position[2] < 0.05:
            reward -= 20
            done = True
        if self.position[2] > 20:
            reward -= 20
            done = True
        if np.abs(self.position[0]) > 15 or np.abs(self.position[1]) > 15:
            reward -= 20
            done = True
        if np.abs(self.orientation[0]) > np.pi*0.8 or np.abs(self.orientation[1]) > np.pi*0.8:
            reward -= 15
            done = True
        if self.current_step >= self.max_steps:
            done = True
        
        self.prev_distance = distance
        return reward, done
    
    def render(self):
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
        pygame.draw.rect(self.screen, (25, 25, 35), (x_offset, y_offset, width, height))
        pygame.draw.rect(self.screen, (60, 60, 80), (x_offset, y_offset, width, height), 2)
        
        title = self.font.render("TOP VIEW - LTL Guided", True, (200, 200, 220))
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
        
        if in_position and is_slow and is_level:
            color = (100, 255, 100)
        elif in_position:
            color = (255, 255, 100)
        else:
            color = (100, 150, 255)
        
        height_ratio = np.clip(self.position[2] / 5.0, 0, 1)
        size = int(8 + 8 * height_ratio)
        
        pygame.draw.circle(self.screen, color, (dx, dy), size)
        pygame.draw.circle(self.screen, (255, 255, 255), (dx, dy), size + 2, 1)
        
        yaw = self.orientation[2]
        arrow_len = size + 15
        ax = int(dx + arrow_len * np.cos(yaw))
        ay = int(dy + arrow_len * np.sin(yaw))
        pygame.draw.line(self.screen, (255, 255, 100), (dx, dy), (ax, ay), 3)
    
    def _render_side_view(self, x_offset, y_offset, width, height):
        pygame.draw.rect(self.screen, (25, 25, 35), (x_offset, y_offset, width, height))
        pygame.draw.rect(self.screen, (60, 60, 80), (x_offset, y_offset, width, height), 2)
        
        title = self.font.render("SIDE VIEW (X-Z)", True, (200, 200, 220))
        self.screen.blit(title, (x_offset + 10, y_offset + 10))
        
        scale = 20
        center_x = x_offset + width // 2
        ground_y = y_offset + height - 30
        
        pygame.draw.line(self.screen, (100, 100, 80), (x_offset, ground_y), (x_offset + width, ground_y), 3)
        
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
        
        tx = int(center_x + self.target[0] * scale)
        tz = int(ground_y - self.target[2] * scale)
        pygame.draw.circle(self.screen, (255, 80, 80), (tx, tz), 10)
        
        dx = int(center_x + self.position[0] * scale)
        dz = int(ground_y - self.position[2] * scale)
        pygame.draw.circle(self.screen, (100, 150, 255), (dx, dz), 10)
    
    def _render_info_panel(self, x_offset, y_offset):
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
        pygame.quit()