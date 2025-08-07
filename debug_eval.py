"""
Debug evaluation script to isolate issues with PPO evaluation
"""

import os
from stable_baselines3 import PPO
from kung_fu_chess_env import KungFuChessEnv, GameMode

def test_model_loading():
    """Test if model can be loaded"""
    print("=== Testing Model Loading ===")
    
    # Check what models exist
    model_dir = "./models/"
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
        print(f"Available models: {models}")
    else:
        print("Models directory not found!")
        return None
    
    # Try to load the final model (most likely to exist)
    model_path = "./models/ppo_kung_fu_chess_best.zip"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            model = PPO.load(model_path)
            print("✓ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return None
    else:
        print(f"Model not found at {model_path}")
        return None

def test_environment():
    """Test if environment works without model"""
    print("\n=== Testing Environment ===")
    
    try:
        env = KungFuChessEnv(mode=GameMode.LIGHTNING, render_mode="human")
        print("✓ Environment created successfully!")
        
        obs, info = env.reset()
        print("✓ Environment reset successfully!")
        print(f"Observation keys: {obs.keys()}")
        print(f"Board shape: {obs['board'].shape}")
        
        # Test a few random actions
        for i in range(5):
            action = env.action_space.sample()
            print(f"Step {i+1}: Taking action {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Reward: {reward:.2f}, Terminated: {terminated}")
            env.render()
            
            if terminated:
                print("  Game ended!")
                break
        
        env.close()
        print("✓ Environment test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_model_prediction(model):
    """Test if model can make predictions"""
    print("\n=== Testing Model Predictions ===")
    
    try:
        env = KungFuChessEnv(mode=GameMode.LIGHTNING)
        obs, info = env.reset()
        
        print("✓ Environment ready for model testing")
        
        # Test model prediction
        action, _states = model.predict(obs, deterministic=True)
        print(f"✓ Model prediction successful: {action}")
        
        # Test if action is valid
        if env.action_space.contains(action):
            print("✓ Action is valid")
        else:
            print(f"✗ Action is invalid: {action}")
        
        # Test environment step with model action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful: reward={reward:.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Model prediction test failed: {e}")
        return False

def test_full_evaluation(model):
    """Test full evaluation with rendering"""
    print("\n=== Testing Full Evaluation ===")
    
    try:
        env = KungFuChessEnv(mode=GameMode.LIGHTNING, render_mode="human")
        obs, info = env.reset()
        
        print("Starting 10-step evaluation with rendering...")
        print("You should see the pygame window with a chess board and pieces moving")
        
        for step in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step+1}: Action={action}, Reward={reward:.2f}")
            
            env.render()
            
            # Add a longer delay to see what's happening
            import time
            time.sleep(1.0)  # 1 second delay
            
            if terminated:
                print("Game ended!")
                break
        
        env.close()
        print("✓ Full evaluation test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Full evaluation test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("PPO Evaluation Diagnostic Tool")
    print("=" * 40)
    
    # Test 1: Model loading
    model = test_model_loading()
    if not model:
        print("\n❌ Cannot proceed without a valid model")
        return
    
    # Test 2: Environment
    if not test_environment():
        print("\n❌ Environment test failed")
        return
    
    # Test 3: Model predictions
    if not test_model_prediction(model):
        print("\n❌ Model prediction test failed")
        return
    
    # Test 4: Full evaluation
    if not test_full_evaluation(model):
        print("\n❌ Full evaluation test failed")
        return
    
    print("\n✅ All tests passed! The evaluation should work.")
    print("If you still have issues, the problem might be with pygame/display settings.")

if __name__ == "__main__":
    main()
