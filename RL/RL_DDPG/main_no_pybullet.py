"""
Training script with OBSTACLES!
Drone must navigate around barriers to reach target.
"""
import numpy as np
from env_enhanced import RealisticDroneEnv  # Now with obstacles!
from ddpg_agent import DDPGAgent
import time

# ========================
# 🎧 CONFIGURATION
# ========================
MODE = "test"  # "train" or "test"
MODEL_PATH = "best_drone_obstacles.pth"
EPISODES = 3000  # More episodes needed for obstacle avoidance
NUM_OBSTACLES = 5  # Number of obstacles in environment
# ========================

def train():
    """Training mode"""
    env = RealisticDroneEnv(num_obstacles=NUM_OBSTACLES, fixed_obstacles=True)
    
    state_dim = 27  # Updated: 15 (original) + 12 (3 obstacles * 4 features)
    action_dim = 4
    
    agent = DDPGAgent(state_dim, action_dim, max_action=1.0)
    
    print("=" * 60)
    print("🚁 DRONE RL TRAINING - With Obstacles!")
    print("=" * 60)
    print(f"State dimension: {state_dim} (includes obstacle info)")
    print(f"Action dimension: {action_dim} (continuous)")
    print(f"Episodes: {EPISODES}")
    print(f"Obstacles per episode: {NUM_OBSTACLES}")
    print("🚧 Drone must learn to avoid obstacles!")
    print("Press Ctrl+C to stop training and save")
    print("=" * 60)
    
    best_reward = -float('inf')
    episode_rewards = []
    collision_count = 0
    success_count = 0
    
    try:
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            steps = 0
            critic_losses = []
            actor_losses = []
            
            done = False
            while not done:
                # Render environment
                if not env.render():
                    raise KeyboardInterrupt
                
                # Select action
                action = agent.select_action(state, add_noise=True)
                
                # Step environment
                next_state, reward, done = env.step(action)
                
                # Store experience
                agent.buffer.append((state, action, reward, next_state, done))
                
                # Train
                critic_loss, actor_loss = agent.train()
                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Track statistics
            episode_rewards.append(total_reward)
            if total_reward > 450:  # Likely reached target (500 - some penalties)
                success_count += 1
            elif total_reward < -150:  # Likely collision
                collision_count += 1
            
            avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
            avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
            
            # Print progress
            avg_reward_10 = np.mean(episode_rewards[-10:])
            avg_reward_50 = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else avg_reward_10
            success_rate = success_count / (episode + 1) * 100
            collision_rate = collision_count / (episode + 1) * 100
            
            print(f"Ep {episode:4d} | "
                  f"Steps: {steps:4d} | "
                  f"R: {total_reward:7.1f} | "
                  f"Avg10: {avg_reward_10:7.1f} | "
                  f"Avg50: {avg_reward_50:7.1f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Collision: {collision_rate:5.1f}% | "
                  f"C: {avg_critic_loss:.3f} | "
                  f"A: {avg_actor_loss:.3f} | "
                  f"σ: {agent.noise_std:.3f}")
            
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(MODEL_PATH)
                print(f"🌟 New best: {best_reward:.1f}")
            
            # Periodic save
            if episode % 100 == 0 and episode > 0:
                agent.save(f"drone_obstacles_ep{episode}.pth")
                print(f"💾 Checkpoint saved at episode {episode}")
                print(f"   Success rate: {success_rate:.1f}%, Collision rate: {collision_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        agent.save("drone_obstacles_interrupted.pth")
    
    finally:
        env.close()
        print(f"\n✔ Training complete!")
        print(f"Best reward: {best_reward:.1f}")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Final success rate: {success_rate:.1f}%")
        print(f"Final collision rate: {collision_rate:.1f}%")

def test():
    """Testing mode"""
    env = RealisticDroneEnv(num_obstacles=NUM_OBSTACLES, fixed_obstacles=True)
    
    state_dim = 27
    action_dim = 4
    
    agent = DDPGAgent(state_dim, action_dim, max_action=1.0)
    
    try:
        agent.load(MODEL_PATH)
        print("✓ Model loaded successfully")
    except:
        print("⚠️ Could not load model, using random policy")
    
    agent.noise_std = 0.0  # No exploration in testing
    
    print("=" * 60)
    print("🚁 DRONE RL TESTING - With Obstacles")
    print("=" * 60)
    print("Watch the drone navigate around obstacles!")
    print("Press Ctrl+C or close window to exit")
    
    successes = 0
    collisions = 0
    
    try:
        episode = 0
        while True:
            state = env.reset()
            total_reward = 0
            steps = 0
            
            done = False
            while not done:
                if not env.render():
                    raise KeyboardInterrupt
                
                action = agent.select_action(state, add_noise=False)
                next_state, reward, done = env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            episode += 1
            
            # Classify outcome
            if total_reward > 450:
                successes += 1
                outcome = "🎯 SUCCESS"
            elif total_reward < -150:
                collisions += 1
                outcome = "💥 COLLISION"
            else:
                outcome = "⏱️ TIMEOUT"
            
            success_rate = successes / episode * 100 if episode > 0 else 0
            collision_rate = collisions / episode * 100 if episode > 0 else 0
            
            print(f"Episode {episode:3d} | "
                  f"Steps: {steps:4d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"{outcome} | "
                  f"Success: {success_rate:.1f}% | "
                  f"Collisions: {collision_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n👋 Testing stopped")
        print(f"Final stats: {successes} successes, {collisions} collisions in {episode} episodes")
    
    finally:
        env.close()

if __name__ == "__main__":
    if MODE == "train":
        train()
    else:
        test()