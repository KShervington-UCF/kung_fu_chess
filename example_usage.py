"""
Example usage of the Kung Fu Chess Gymnasium Environment
"""

import numpy as np
from kung_fu_chess_env import KungFuChessEnv, GameMode


def random_agent_demo():
    """Demonstrate the environment with random actions"""
    print("=== Kung Fu Chess Environment Demo ===")
    
    # Create environment
    env = KungFuChessEnv(mode=GameMode.LIGHTNING, render_mode="human")
    
    # Reset environment
    observation, info = env.reset()
    print(f"Initial observation keys: {observation.keys()}")
    print(f"Board shape: {observation['board'].shape}")
    print(f"Cooldowns shape: {observation['cooldowns'].shape}")
    
    step_count = 0
    total_reward = 0
    
    try:
        while not env.game_over and step_count < 1000:
            # Sample random action
            action = env.action_space.sample()
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render
            env.render()
            
            if step_count % 100 == 0:
                print(f"Step {step_count}, Total Reward: {total_reward:.2f}")
            
            if terminated:
                print(f"Game Over! Winner: {info.get('winner')}")
                break
                
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    
    finally:
        env.close()
        print(f"Demo completed. Total steps: {step_count}, Total reward: {total_reward:.2f}")


def test_basic_functionality():
    """Test basic environment functionality"""
    print("=== Testing Basic Functionality ===")
    
    # Test both game modes
    for mode in [GameMode.STANDARD, GameMode.LIGHTNING]:
        print(f"\nTesting {mode.value} mode:")
        
        env = KungFuChessEnv(mode=mode)
        
        # Test reset
        obs, info = env.reset()
        assert obs['board'].shape == (8, 8)
        assert obs['cooldowns'].shape == (32,)
        assert obs['current_player'] in [0, 1]
        print(f"✓ Reset successful")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step successful, reward: {reward}")
        
        # Test observation space
        assert env.observation_space.contains(obs)
        print(f"✓ Observation space validation passed")
        
        # Test action space
        assert env.action_space.contains(action)
        print(f"✓ Action space validation passed")
        
        env.close()
    
    print("\n=== All tests passed! ===")


def demonstrate_piece_movement():
    """Demonstrate specific piece movements"""
    print("=== Demonstrating Piece Movement ===")
    
    env = KungFuChessEnv(mode=GameMode.LIGHTNING)
    obs, info = env.reset()
    
    print("Initial board state:")
    print(obs['board'])
    
    # Try to move a pawn (piece index 8-15 are typically pawns)
    pawn_action = [8, 4, 0]  # Move pawn to row 4, col 0
    obs, reward, terminated, truncated, info = env.step(pawn_action)
    print(f"Pawn move reward: {reward}")
    
    # Try to move a rook (piece index 0 or 7)
    rook_action = [0, 5, 0]  # Move rook to row 5, col 0
    obs, reward, terminated, truncated, info = env.step(rook_action)
    print(f"Rook move reward: {reward}")
    
    print("Board after moves:")
    print(obs['board'])
    
    env.close()


if __name__ == "__main__":
    # Run tests first
    test_basic_functionality()
    
    # Demonstrate piece movement
    demonstrate_piece_movement()
    
    # Run interactive demo (comment out if running headless)
    # random_agent_demo()
