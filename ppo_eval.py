"""
PPO Evaluation Script for Kung Fu Chess Environment

This script evaluates trained PPO agents with comprehensive metrics,
statistics, and visualization options.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass

from sb3_contrib import MaskablePPO
from kung_fu_chess_env import KungFuChessEnv, GameMode, Color


@dataclass
class GameStats:
    """Statistics for a single game"""
    total_reward: float
    game_length: int
    winner: Color
    capture_reward: float
    positional_reward: float
    win_reward: float
    pieces_captured: int
    game_time: float


class PPOEvaluator:
    """Evaluator for trained PPO agents"""
    
    def __init__(self, model_path: str, game_mode: GameMode = GameMode.LIGHTNING):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            game_mode: Game mode for evaluation
        """
        self.model_path = model_path
        self.game_mode = game_mode
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        self.model = MaskablePPO.load(self.model_path)
        print("Model loaded successfully")
    
    def evaluate_single_game(
        self, 
        render: bool = False, 
        max_steps: int = 1000,
        deterministic: bool = True
    ) -> GameStats:
        """Evaluate a single game and return statistics"""
        
        render_mode = "human" if render else None
        env = KungFuChessEnv(mode=self.game_mode, render_mode=render_mode)
        
        obs, info = env.reset()
        total_reward = 0.0
        capture_reward_sum = 0.0
        positional_reward_sum = 0.0
        win_reward_sum = 0.0
        step_count = 0
        
        start_time = time.time()
        
        for step in range(max_steps):
            action, _states = self.model.predict(obs, deterministic=deterministic)

            # Perform random action || THIS IS FOR DEBUGGING
            # action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)


            # Print action and reward for debugging
            print(f"Step {step}: Action={action}, Reward={reward:.2f}")
            
            total_reward += reward
            capture_reward_sum += info.get('capture_reward', 0)
            positional_reward_sum += info.get('positional_reward', 0)
            win_reward_sum += info.get('win_reward', 0)
            step_count += 1
            
            if render:
                env.render()
                time.sleep(0.1)  # Slow down for human viewing
            
            if terminated or truncated:
                break
        
        game_time = time.time() - start_time
        winner = info.get('winner')
        pieces_captured = len(env.captured_pieces)
        
        env.close()
        
        return GameStats(
            total_reward=total_reward,
            game_length=step_count,
            winner=winner,
            capture_reward=capture_reward_sum,
            positional_reward=positional_reward_sum,
            win_reward=win_reward_sum,
            pieces_captured=pieces_captured,
            game_time=game_time
        )
    
    def evaluate_multiple_games(
        self, 
        n_games: int = 10, 
        deterministic: bool = True
    ) -> List[GameStats]:
        """Evaluate multiple games and return statistics"""
        
        print(f"Evaluating {n_games} games...")
        stats_list = []
        
        for game_num in range(n_games):
            if (game_num + 1) % 10 == 0:
                print(f"Completed {game_num + 1}/{n_games} games")
            
            stats = self.evaluate_single_game(
                render=False, 
                deterministic=deterministic
            )
            stats_list.append(stats)
        
        return stats_list
    
    def analyze_performance(self, stats_list: List[GameStats]) -> Dict:
        """Analyze performance across multiple games"""
        
        if not stats_list:
            return {}
        
        # Extract metrics
        rewards = [s.total_reward for s in stats_list]
        game_lengths = [s.game_length for s in stats_list]
        capture_rewards = [s.capture_reward for s in stats_list]
        positional_rewards = [s.positional_reward for s in stats_list]
        win_rewards = [s.win_reward for s in stats_list]
        pieces_captured = [s.pieces_captured for s in stats_list]
        game_times = [s.game_time for s in stats_list]
        
        # Count wins/losses
        wins = sum(1 for s in stats_list if s.winner == Color.WHITE)  # Assuming agent is WHITE
        losses = sum(1 for s in stats_list if s.winner == Color.BLACK)
        draws = len(stats_list) - wins - losses
        
        analysis = {
            'total_games': len(stats_list),
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / len(stats_list) if stats_list else 0,
            
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards)
            },
            
            'game_length_stats': {
                'mean': np.mean(game_lengths),
                'std': np.std(game_lengths),
                'min': np.min(game_lengths),
                'max': np.max(game_lengths),
                'median': np.median(game_lengths)
            },
            
            'capture_stats': {
                'mean_reward': np.mean(capture_rewards),
                'mean_pieces': np.mean(pieces_captured),
                'total_pieces': np.sum(pieces_captured)
            },
            
            'positional_reward_mean': np.mean(positional_rewards),
            'win_reward_mean': np.mean(win_rewards),
            'avg_game_time': np.mean(game_times)
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print formatted analysis results"""
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Total Games: {analysis['total_games']}")
        print(f"Wins: {analysis['wins']} ({analysis['win_rate']:.1%})")
        print(f"Losses: {analysis['losses']}")
        print(f"Draws: {analysis['draws']}")
        
        print("\nReward Statistics:")
        reward_stats = analysis['reward_stats']
        print(f"  Mean: {reward_stats['mean']:.2f} ± {reward_stats['std']:.2f}")
        print(f"  Range: [{reward_stats['min']:.2f}, {reward_stats['max']:.2f}]")
        print(f"  Median: {reward_stats['median']:.2f}")
        
        print("\nGame Length Statistics:")
        length_stats = analysis['game_length_stats']
        print(f"  Mean: {length_stats['mean']:.1f} ± {length_stats['std']:.1f} steps")
        print(f"  Range: [{length_stats['min']}, {length_stats['max']}] steps")
        print(f"  Median: {length_stats['median']:.1f} steps")
        
        print("\nCapture Statistics:")
        capture_stats = analysis['capture_stats']
        print(f"  Mean capture reward: {capture_stats['mean_reward']:.2f}")
        print(f"  Mean pieces captured per game: {capture_stats['mean_pieces']:.1f}")
        print(f"  Total pieces captured: {capture_stats['total_pieces']}")
        
        print(f"\nMean positional reward: {analysis['positional_reward_mean']:.2f}")
        print(f"Mean win reward: {analysis['win_reward_mean']:.2f}")
        print(f"Average game time: {analysis['avg_game_time']:.2f} seconds")
    
    def plot_performance(self, stats_list: List[GameStats], save_path: str = None):
        """Plot performance metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PPO Agent Performance Analysis', fontsize=16)
        
        # Reward distribution
        rewards = [s.total_reward for s in stats_list]
        axes[0, 0].hist(rewards, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Total Reward Distribution')
        axes[0, 0].set_xlabel('Total Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        axes[0, 0].legend()
        
        # Game length distribution
        lengths = [s.game_length for s in stats_list]
        axes[0, 1].hist(lengths, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Game Length Distribution')
        axes[0, 1].set_xlabel('Game Length (steps)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
        axes[0, 1].legend()
        
        # Reward components
        capture_rewards = [s.capture_reward for s in stats_list]
        positional_rewards = [s.positional_reward for s in stats_list]
        axes[1, 0].scatter(capture_rewards, positional_rewards, alpha=0.6)
        axes[1, 0].set_title('Capture vs Positional Rewards')
        axes[1, 0].set_xlabel('Capture Reward')
        axes[1, 0].set_ylabel('Positional Reward')
        
        # Win/Loss pie chart
        wins = sum(1 for s in stats_list if s.winner == Color.WHITE)
        losses = sum(1 for s in stats_list if s.winner == Color.BLACK)
        draws = len(stats_list) - wins - losses
        
        labels = ['Wins', 'Losses', 'Draws']
        sizes = [wins, losses, draws]
        colors = ['green', 'red', 'yellow']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Game Outcomes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to {save_path}")
        
        plt.show()


def main():
    """Main evaluation function"""
    
    # Configuration
    MODEL_PATH = "./models/ppo_kung_fu_chess_final.zip"  # Use best model
    GAME_MODE = GameMode.LIGHTNING
    N_EVAL_GAMES = 100
    
    try:
        # Create evaluator
        evaluator = PPOEvaluator(MODEL_PATH, GAME_MODE)
        
        # Option 1: Single game with visualization
        print("Running single game with visualization...")
        single_stats = evaluator.evaluate_single_game(render=True, max_steps=1000, deterministic=False)
        print(f"Single game result: Reward={single_stats.total_reward:.2f}, Winner={single_stats.winner}")
        
        # Option 2: Multiple games for statistics
        # print(f"\nRunning {N_EVAL_GAMES} games for statistical analysis...")
        # stats_list = evaluator.evaluate_multiple_games(N_EVAL_GAMES, deterministic=True)
        
        # # Analyze and print results
        # analysis = evaluator.analyze_performance(stats_list)
        # evaluator.print_analysis(analysis)
        
        # # Plot results
        # evaluator.plot_performance(stats_list, "./evaluation_results/evaluation_results.png")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to train a model first using ppo_train.py")
    except Exception as e:
        print(f"Evaluation error: {e}")


if __name__ == "__main__":
    main()