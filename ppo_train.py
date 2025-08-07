"""
PPO Training Script for Kung Fu Chess Environment

This script trains a PPO agent on the Kung Fu Chess environment with proper
hyperparameters, logging, and checkpointing for effective RL training.
"""

import os
import time
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from kung_fu_chess_env import KungFuChessEnv, GameMode


def make_env(game_mode: GameMode = GameMode.LIGHTNING, rank: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    Args:
        game_mode: Game mode (STANDARD or LIGHTNING)
        rank: Index of the subprocess
    """
    def _init() -> KungFuChessEnv:
        env = KungFuChessEnv(mode=game_mode)
        env = Monitor(env, f"./logs/train_env_{rank}")
        return env
    return _init


def train_ppo_agent(
    total_timesteps: int = 100_000,
    game_mode: GameMode = GameMode.LIGHTNING,
    n_envs: int = 4,
    save_path: str = "./models/ppo_kung_fu_chess",
    log_path: str = "./logs/"
):
    """
    Train a PPO agent on Kung Fu Chess environment.
    
    Args:
        total_timesteps: Total training timesteps
        game_mode: Game mode to train on
        n_envs: Number of parallel environments
        save_path: Path to save the trained model
        log_path: Path for logging
    """
    
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    print(f"Training PPO agent on Kung Fu Chess ({game_mode.value} mode)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Environment type: DummyVecEnv (WSL compatible)")
    
    # Create vectorized environment
    # Use DummyVecEnv for WSL compatibility (avoids multiprocessing issues)
    env = DummyVecEnv([make_env(game_mode, i) for i in range(n_envs)])
    
    # Create evaluation environment (same type as training env)
    # eval_env = DummyVecEnv([make_env(game_mode, 999)])
    
    # PPO hyperparameters optimized for complex environments
    model = PPO(
        "MultiInputPolicy",
        env,
        # Learning rate with schedule
        learning_rate=3e-4,
        # PPO-specific parameters
        n_steps=2048,  # Steps per environment per update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of epochs per update
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clip range
        clip_range_vf=None,  # Value function clip range
        ent_coef=0.01,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        # Network architecture
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log=log_path,
        device='auto'
    )
    
    # Callbacks for monitoring and checkpointing
    # eval_callback = EvalCallback(
    #     # eval_env,
    #     best_model_save_path=save_path + "_best",
    #     log_path=log_path,
    #     eval_freq=25000,  # Evaluate every 25k steps (less frequent for WSL)
    #     n_eval_episodes=5,   # Fewer episodes for faster evaluation
    #     deterministic=True,
    #     render=False,
    #     verbose=1  # Add verbose output for debugging
    # )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # Number of timesteps between checkpoints
        save_path=save_path + "_checkpoints",
        name_prefix="ppo_kung_fu_chess"
    )
    
    # callback_list = CallbackList([eval_callback, checkpoint_callback])
    callback_list = CallbackList([checkpoint_callback])
    
    # Train the agent
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name="PPO_KungFuChess"
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        model.save(save_path + "_final")
        print(f"Model saved to {save_path}_final.zip")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        model.save(save_path + "_interrupted")
        print(f"Model saved to {save_path}_interrupted.zip")
    
    finally:
        env.close()
        # eval_env.close()


if __name__ == "__main__":
    # Training configuration
    TOTAL_TIMESTEPS = 1_000_000  # 1M timesteps for good learning
    GAME_MODE = GameMode.LIGHTNING  # Fast-paced training
    N_ENVS = 4  # Parallel environments for faster training
    
    # Start training
    train_ppo_agent(
        total_timesteps=TOTAL_TIMESTEPS,
        game_mode=GAME_MODE,
        n_envs=N_ENVS
    )