"""
DDPG (Deep Deterministic Policy Gradient) Agent
Better than DQN for continuous control like drones!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    """Policy network that outputs actions directly"""
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):
    """Value network that evaluates state-action pairs"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, 
                 use_hover_bias=False, hover_bias_decay=0.9995):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ðŸ–¥ï¸  Using device: {self.device}")
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Replay buffer
        self.buffer = deque(maxlen=100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter
        self.batch_size = 128
        self.noise_std = 0.3  # Increased initial exploration
        self.noise_decay = 0.99995  # Slower decay
        self.noise_min = 0.05
        
        self.action_dim = action_dim
        self.max_action = max_action
        # Hover bias system (optional training wheels)
        self.use_hover_bias = use_hover_bias
        self.hover_bias_weight = 0.3 if use_hover_bias else 0.0
        self.hover_bias_decay = hover_bias_decay
        self.hover_bias_min = 0.0
        print(f"ðŸ“Š State dim: {state_dim}, Action dim: {action_dim}")
        print("ðŸŽ¯ Using DDPG (continuous control)")
    
    def select_action(self, state, add_noise=True):
        """Select action with optional exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, 0, self.max_action)
        
        # Optional hover bias - helps early training but should decay
        if self.hover_bias_weight > self.hover_bias_min:
            hover_thrust = 0.6
            action = action * (1.0 - self.hover_bias_weight) + hover_thrust * self.hover_bias_weight
            self.hover_bias_weight *= self.hover_bias_decay
        
        return action
    
    def train(self):
        """Train actor and critic networks"""
        if len(self.buffer) < self.batch_size:
            return None, None
        
        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ====== Train Critic ======
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ====== Train Actor ======
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ====== Soft update target networks ======
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # Decay noise
        self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source, target):
        """Soft update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, path="drone_ddpg.pth"):
        """Save models"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_std': self.noise_std,
            'hover_bias_weight': self.hover_bias_weight
        }, path)
        print(f"ðŸ’¾ DDPG model saved to {path}")
    
    def load(self, path="drone_ddpg.pth"):
        """Load models"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.noise_std = checkpoint.get('noise_std', 0.0)
        self.hover_bias_weight = checkpoint.get('hover_bias_weight', 0.0)
        print(f"ðŸ“‚ DDPG model loaded from {path}")