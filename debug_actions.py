"""
Debug script to analyze what actions the agent is taking and why they're invalid
"""

from stable_baselines3 import PPO
from kung_fu_chess_env import KungFuChessEnv, GameMode

def debug_agent_actions():
    """Debug what actions the agent is taking and why they fail"""
    
    # Load model
    model = PPO.load("./models/ppo_kung_fu_chess_best.zip")
    env = KungFuChessEnv(mode=GameMode.LIGHTNING)
    
    obs, info = env.reset()
    
    print("=== Agent Action Analysis ===")
    print("Initial board state:")
    print(obs['board'])
    print()
    
    # Print initial piece positions
    print("White pieces:")
    for i, piece in enumerate(env.pieces[env.current_player]):
        print(f"  Piece {i}: {piece.piece_type.name} at {piece.position}")
    print()
    
    for step in range(20):  # Analyze first 20 actions
        action, _states = model.predict(obs, deterministic=True)
        piece_idx, target_row, target_col = action
        target_pos = (target_row, target_col)
        
        print(f"Step {step+1}:")
        print(f"  Action: piece_idx={piece_idx}, target=({target_row},{target_col})")
        
        # Get current player's pieces
        current_pieces = env.pieces[env.current_player]
        
        # Check if piece index is valid
        if piece_idx >= len(current_pieces):
            print(f"  ❌ Invalid piece index {piece_idx} (only {len(current_pieces)} pieces)")
        else:
            piece = current_pieces[piece_idx]
            print(f"  Piece: {piece.piece_type.name} at {piece.position}")
            
            # Check why move is invalid
            if not env._is_on_board(target_pos):
                print(f"  ❌ Target {target_pos} is off the board")
            elif piece.position == target_pos:
                print(f"  ❌ Trying to move to same position {target_pos}")
            elif not env._is_valid_move(piece, target_pos):
                print(f"  ❌ Invalid move for {piece.piece_type.name}: {piece.position} -> {target_pos}")
                
                # More specific diagnosis
                row_diff = target_pos[0] - piece.position[0]
                col_diff = target_pos[1] - piece.position[1]
                print(f"    Row diff: {row_diff}, Col diff: {col_diff}")
                
                if piece.piece_type.name == "PAWN":
                    print(f"    Pawn move rules violated")
                elif piece.piece_type.name == "ROOK":
                    if row_diff != 0 and col_diff != 0:
                        print(f"    Rook must move in straight line")
                # Add more piece-specific diagnostics as needed
            else:
                print(f"  ✅ Valid move")
        
        # Take the step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Reward: {reward}")
        print()
        
        if terminated:
            print("Game ended!")
            break
    
    env.close()

if __name__ == "__main__":
    debug_agent_actions()
