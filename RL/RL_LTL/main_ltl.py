"""
Training script for drone RL with LTL specifications
Fixed physics, proper rotation matrices, better reward structure

Usage:
  - Set MODE = "train" to train from scratch
  - Set MODE = "test" to evaluate a trained model
  - Press Ctrl+C to stop and save
"""
import numpy as np
from env_enhanced_ltl import RealisticDroneEnvLTL
from ddpg_agent import DDPGAgent
from ltl_specifications import create_drone_ltl_monitor
import time

# ========================
# 🎧 CONFIGURATION
# ========================
MODE = "test"  # "train" or "test"
MODEL_PATH = "best_drone_ltl.pth"
EPISODES = 3000
NUM_OBSTACLES = 2

# Training options
USE_HOVER_BIAS = False  # Set True for easier initial training (training wheels)
HOVER_BIAS_DECAY = 0.9995  # How fast to remove hover assistance
# ========================

def train():
    """
    Main training loop
    
    Creates environment with LTL monitor, trains DDPG agent
    Saves best model and periodic checkpoints
    """
    # Setup environment
    # Need to create temp env first to pass to LTL monitor
    # (chicken-and-egg problem since monitor needs env reference)
    temp_env = RealisticDroneEnvLTL(num_obstacles=NUM_OBSTACLES, fixed_obstacles=True)
    
    # Create LTL monitor with all the formal specifications
    ltl_monitor = create_drone_ltl_monitor(temp_env)
    
    # Now create actual environment with monitor attached
    env = RealisticDroneEnvLTL(
        num_obstacles=NUM_OBSTACLES, 
        fixed_obstacles=True,
        ltl_monitor=ltl_monitor
    )
    
    # Setup agent
    state_dim = 27  # see _get_state() for breakdown
    action_dim = 4  # 4 motor thrusts
    
    agent = DDPGAgent(
        state_dim, 
        action_dim, 
        max_action=1.0,
        use_hover_bias=USE_HOVER_BIAS,
        hover_bias_decay=HOVER_BIAS_DECAY
    )
    
    # Print training info
    print("=" * 70)
    print("🚁 DRONE RL TRAINING - With LTL Specifications")
    print("=" * 70)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Episodes: {EPISODES}")
    print(f"Obstacles: {NUM_OBSTACLES}")
    print(f"Hover bias: {USE_HOVER_BIAS}")
    print("\n📋 LTL SPECIFICATIONS:")
    print("-" * 70)
    
    for spec in ltl_monitor.specifications:
        print(f"  [{spec.property_type.value.upper()}] {spec.name}")
        print(f"    → {spec.description}")
    
    print("=" * 70)
    print("Press Ctrl+C to stop training and save")
    print("=" * 70)
    
    # Training stats
    best_reward = -float('inf')
    episode_rewards = []
    base_rewards = []
    ltl_rewards = []
    success_count = 0
    collision_count = 0
    ltl_compliant_episodes = 0
    
    try:
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            total_base_reward = 0
            total_ltl_reward = 0
            steps = 0
            critic_losses = []
            actor_losses = []
            episode_violations = []
            episode_satisfactions = []
            
            done = False
            while not done:
                # Render (returns False if window closed)
                if not env.render():
                    raise KeyboardInterrupt
                
                # Select action (with exploration noise during training)
                action = agent.select_action(state, add_noise=True)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience in replay buffer
                agent.buffer.append((state, action, reward, next_state, done))
                
                # Train on batch from replay buffer
                critic_loss, actor_loss = agent.train()
                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                
                # Move to next state
                state = next_state
                total_reward += reward
                total_base_reward += info['base_reward']
                total_ltl_reward += info['ltl_reward']
                
                # Track LTL violations and satisfactions
                if info['violations']:
                    episode_violations.extend(info['violations'])
                if info['satisfactions']:
                    episode_satisfactions.extend(info['satisfactions'])
                
                steps += 1
            
            # Episode finished - compute statistics
            episode_rewards.append(total_reward)
            base_rewards.append(total_base_reward)
            ltl_rewards.append(total_ltl_reward)
            
            # Classify episode outcome
            if total_reward > 500:
                success_count += 1
            elif total_reward < -200:
                collision_count += 1
            
            # Check if episode was LTL compliant (no critical safety violations)
            critical_violations = [v for v in episode_violations 
                                 if 'collision' in v or 'bounds' in v or 'flip' in v]
            if not critical_violations:
                ltl_compliant_episodes += 1
            
            # Compute moving averages
            avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
            avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
            
            avg_reward_10 = np.mean(episode_rewards[-10:])
            avg_reward_50 = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else avg_reward_10
            avg_base_10 = np.mean(base_rewards[-10:])
            avg_ltl_10 = np.mean(ltl_rewards[-10:])
            
            # Compute rates
            success_rate = success_count / (episode + 1) * 100
            collision_rate = collision_count / (episode + 1) * 100
            compliance_rate = ltl_compliant_episodes / (episode + 1) * 100
            
            # Print episode summary
            print(f"\nEp {episode:4d} | Steps: {steps:4d}")
            print(f"  Total: {total_reward:7.1f} (Base: {total_base_reward:6.1f}, LTL: {total_ltl_reward:6.1f})")
            print(f"  Avg10: {avg_reward_10:7.1f} (Base: {avg_base_10:6.1f}, LTL: {avg_ltl_10:6.1f})")
            print(f"  Avg50: {avg_reward_50:7.1f}")
            print(f"  Success: {success_rate:5.1f}% | Collision: {collision_rate:5.1f}% | Compliance: {compliance_rate:5.1f}%")
            print(f"  Losses - C: {avg_critic_loss:.3f}, A: {avg_actor_loss:.3f} | σ: {agent.noise_std:.3f}")
            
            # Show violations/satisfactions
            if episode_violations:
                unique_viol = list(set(episode_violations))[:3]
                print(f"  ⚠️  Violations: {', '.join(unique_viol)}")
            if episode_satisfactions:
                unique_sats = list(set(episode_satisfactions))[:3]
                print(f"  ✅ Satisfied: {', '.join(unique_sats)}")
            
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(MODEL_PATH)
                print(f"  🌟 New best: {best_reward:.1f}")
            
            # Periodic checkpoints and statistics
            if episode % 100 == 0 and episode > 0:
                agent.save(f"drone_ltl_ep{episode}.pth")
                print(f"\n{'='*70}")
                print(f"💾 Checkpoint saved at episode {episode}")
                print(f"{'='*70}")
                ltl_monitor.print_statistics()
                print(f"{'='*70}\n")
    
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        agent.save("drone_ltl_interrupted.pth")
    
    finally:
        env.close()
        print(f"\n{'='*70}")
        print("✓ Training complete!")
        print("="*70)
        print(f"Best reward: {best_reward:.1f}")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Final success rate: {success_rate:.1f}%")
        print(f"Final collision rate: {collision_rate:.1f}%")
        print(f"Final LTL compliance rate: {compliance_rate:.1f}%")
        print("="*70)
        
        # Print final LTL statistics
        ltl_monitor.print_statistics()

def test():
    """
    Testing/evaluation mode
    
    Loads trained model and runs episodes without exploration noise
    Good for visualizing learned behavior and checking LTL compliance
    """
    # Setup environment (same as training)
    temp_env = RealisticDroneEnvLTL(num_obstacles=NUM_OBSTACLES, fixed_obstacles=True)
    ltl_monitor = create_drone_ltl_monitor(temp_env)
    
    env = RealisticDroneEnvLTL(
        num_obstacles=NUM_OBSTACLES,
        fixed_obstacles=True,
        ltl_monitor=ltl_monitor
    )
    
    # Setup agent
    state_dim = 27
    action_dim = 4
    
    agent = DDPGAgent(state_dim, action_dim, max_action=1.0)
    
    # Load trained model
    try:
        agent.load(MODEL_PATH)
        print("✓ Model loaded successfully")
    except:
        print("⚠️  Could not load model, using random policy")
    
    # No exploration noise during testing
    agent.noise_std = 0.0
    
    print("=" * 70)
    print("🚁 DRONE RL TESTING - With LTL Specifications")
    print("=" * 70)
    print("\n📋 LTL SPECIFICATIONS BEING MONITORED:")
    print("-" * 70)
    
    for spec in ltl_monitor.specifications:
        print(f"  [{spec.property_type.value.upper()}] {spec.name}")
    
    print("=" * 70)
    print("Press Ctrl+C or close window to exit")
    print("=" * 70)
    
    # Testing stats
    successes = 0
    collisions = 0
    compliant_episodes = 0
    total_base_reward = 0
    total_ltl_reward = 0
    
    try:
        episode = 0
        while True:
            state = env.reset()
            ep_total_reward = 0
            ep_base_reward = 0
            ep_ltl_reward = 0
            steps = 0
            episode_violations = []
            episode_satisfactions = []
            
            done = False
            while not done:
                # Render
                if not env.render():
                    raise KeyboardInterrupt
                
                # Select action (no noise)
                action = agent.select_action(state, add_noise=False)
                
                # Step
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                ep_total_reward += reward
                ep_base_reward += info['base_reward']
                ep_ltl_reward += info['ltl_reward']
                
                # Track LTL info
                if info['violations']:
                    episode_violations.extend(info['violations'])
                if info['satisfactions']:
                    episode_satisfactions.extend(info['satisfactions'])
                
                steps += 1
            
            episode += 1
            total_base_reward += ep_base_reward
            total_ltl_reward += ep_ltl_reward
            
            # Classify outcome
            if ep_total_reward > 500:
                successes += 1
                outcome = "🎯 SUCCESS"
            elif ep_total_reward < -200:
                collisions += 1
                outcome = "💥 COLLISION"
            else:
                outcome = "⏱️ TIMEOUT"
            
            # Check compliance
            critical_violations = [v for v in episode_violations 
                                 if 'collision' in v or 'bounds' in v or 'flip' in v]
            if not critical_violations:
                compliant_episodes += 1
            
            # Compute rates
            success_rate = successes / episode * 100 if episode > 0 else 0
            collision_rate = collisions / episode * 100 if episode > 0 else 0
            compliance_rate = compliant_episodes / episode * 100 if episode > 0 else 0
            avg_base = total_base_reward / episode
            avg_ltl = total_ltl_reward / episode
            
            # Print summary
            print(f"\nEpisode {episode:3d} | Steps: {steps:4d}")
            print(f"  Reward: {ep_total_reward:7.1f} (Base: {ep_base_reward:6.1f}, LTL: {ep_ltl_reward:6.1f})")
            print(f"  {outcome}")
            print(f"  Success: {success_rate:.1f}% | Collisions: {collision_rate:.1f}% | Compliance: {compliance_rate:.1f}%")
            print(f"  Avg Base: {avg_base:.1f} | Avg LTL: {avg_ltl:.1f}")
            
            if episode_violations:
                unique_viol = list(set(episode_violations))
                print(f"  ⚠️  Violations: {', '.join(unique_viol)}")
            if episode_satisfactions:
                unique_sats = list(set(episode_satisfactions))[:3]
                print(f"  ✅ Satisfied: {', '.join(unique_sats)}")
            
            # Print statistics every 10 episodes
            if episode % 10 == 0:
                print("\n" + "="*70)
                ltl_monitor.print_statistics()
                print("="*70)
    
    except KeyboardInterrupt:
        print("\n👋 Testing stopped")
        print(f"\nFinal Statistics:")
        print(f"  Episodes: {episode}")
        print(f"  Successes: {successes}")
        print(f"  Collisions: {collisions}")
        print(f"  LTL Compliant: {compliant_episodes}")
        print("\n" + "="*70)
        ltl_monitor.print_statistics()
    
    finally:
        env.close()

if __name__ == "__main__":
    if MODE == "train":
        train()
    else:
        test()