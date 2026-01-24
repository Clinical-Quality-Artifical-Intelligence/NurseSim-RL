"""
Train a PPO Agent on the Semantic Triage Environment.

This script trains an agent that learns from NurseEmbed-encoded clinical observations.
"""

import os
import numpy as np
from datetime import datetime

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Our environment
from nursesim_rl import TriageEnv, NurseEmbedWrapper


class PrintProgressCallback(BaseCallback):
    """Simple callback to print training progress."""
    
    def __init__(self, print_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            # Get recent episode rewards if available
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                print(f"Step {self.n_calls}: Mean Reward = {mean_reward:.2f}, Mean Length = {mean_length:.1f}")
        return True


def make_semantic_env():
    """Factory function for creating the semantic environment."""
    base_env = TriageEnv(max_steps=50, max_patients=20)
    semantic_env = NurseEmbedWrapper(base_env, use_vitals=True)
    return semantic_env


def train_semantic_agent(
    total_timesteps: int = 50000,
    save_path: str = "models/semantic_ppo",
    log_dir: str = "logs/semantic_ppo",
):
    """Train a PPO agent on the semantic triage environment."""
    
    print("=" * 60)
    print("SEMANTIC RL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized environment
    print("\n[1] Creating Semantic Environment...")
    env = DummyVecEnv([make_semantic_env])
    
    print(f"    Observation space: {env.observation_space}")
    print(f"    Action space: {env.action_space}")
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_semantic_env])
    
    # Create the PPO agent
    print("\n[2] Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=log_dir,
    )
    
    print(f"    Policy architecture: {model.policy}")
    
    # Callbacks
    progress_callback = PrintProgressCallback(print_freq=2000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )
    
    # Train!
    print(f"\n[3] Training for {total_timesteps:,} timesteps...")
    print("-" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, eval_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_path = os.path.join(save_path, "semantic_ppo_final")
    model.save(final_path)
    print(f"\n[4] Model saved to: {final_path}")
    
    # Quick evaluation
    print("\n[5] Final Evaluation (10 episodes)...")
    rewards = []
    for i in range(10):
        obs = eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
        rewards.append(episode_reward)
    
    print(f"    Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"    Best Episode: {np.max(rewards):.2f}")
    print(f"    Worst Episode: {np.min(rewards):.2f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # Run with reduced timesteps for quick demo
    train_semantic_agent(total_timesteps=20000)
