"""
Unit tests for the Kung Fu Chess Gymnasium Environment
"""

import unittest
import numpy as np
import time
from kung_fu_chess_env import KungFuChessEnv, GameMode, PieceType, Color, Piece


class TestKungFuChessEnv(unittest.TestCase):
    """Test cases for the Kung Fu Chess environment"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = KungFuChessEnv(mode=GameMode.LIGHTNING)
    
    def tearDown(self):
        """Clean up after tests"""
        self.env.close()
    
    def test_initialization(self):
        """Test environment initialization"""
        # Test action space
        self.assertEqual(self.env.action_space.nvec.tolist(), [16, 8, 8])
        
        # Test observation space
        self.assertIn('board', self.env.observation_space.spaces)
        self.assertIn('cooldowns', self.env.observation_space.spaces)
        self.assertIn('current_player', self.env.observation_space.spaces)
        self.assertIn('game_time', self.env.observation_space.spaces)
        
        # Test board dimensions
        self.assertEqual(self.env.board_size, 8)
    
    def test_reset(self):
        """Test environment reset"""
        obs, info = self.env.reset()
        
        # Check observation structure
        self.assertIsInstance(obs, dict)
        self.assertEqual(obs['board'].shape, (8, 8))
        self.assertEqual(obs['cooldowns'].shape, (32,))
        self.assertIn(obs['current_player'], [0, 1])
        
        # Check initial piece count
        white_pieces = len(self.env.pieces[Color.WHITE])
        black_pieces = len(self.env.pieces[Color.BLACK])
        self.assertEqual(white_pieces, 16)
        self.assertEqual(black_pieces, 16)
        
        # Check game state
        self.assertFalse(self.env.game_over)
        self.assertIsNone(self.env.winner)
    
    def test_piece_setup(self):
        """Test initial piece setup"""
        self.env.reset()
        
        # Check white pieces positions
        white_pieces = self.env.pieces[Color.WHITE]
        
        # Check back rank pieces
        back_rank_pieces = [p for p in white_pieces if p.position[0] == 7]
        self.assertEqual(len(back_rank_pieces), 8)
        
        # Check pawns
        pawn_pieces = [p for p in white_pieces if p.position[0] == 6]
        self.assertEqual(len(pawn_pieces), 8)
        
        # Check black pieces positions
        black_pieces = self.env.pieces[Color.BLACK]
        
        # Check back rank pieces
        back_rank_pieces = [p for p in black_pieces if p.position[0] == 0]
        self.assertEqual(len(back_rank_pieces), 8)
        
        # Check pawns
        pawn_pieces = [p for p in black_pieces if p.position[0] == 1]
        self.assertEqual(len(pawn_pieces), 8)
    
    def test_valid_moves(self):
        """Test move validation"""
        self.env.reset()
        
        # Create test pieces
        pawn = Piece(PieceType.PAWN, Color.WHITE, (6, 0))
        rook = Piece(PieceType.ROOK, Color.WHITE, (7, 0))
        knight = Piece(PieceType.KNIGHT, Color.WHITE, (7, 1))
        
        # Test pawn moves
        self.assertTrue(self.env._is_valid_move(pawn, (5, 0)))  # One forward
        self.assertTrue(self.env._is_valid_move(pawn, (4, 0)))  # Two forward from start
        self.assertFalse(self.env._is_valid_move(pawn, (5, 1)))  # Diagonal without capture
        
        # Test rook moves
        self.assertTrue(self.env._is_valid_move(rook, (7, 7)))  # Horizontal
        self.assertTrue(self.env._is_valid_move(rook, (0, 0)))  # Vertical
        self.assertFalse(self.env._is_valid_move(rook, (6, 1)))  # Diagonal
        
        # Test knight moves
        self.assertTrue(self.env._is_valid_move(knight, (5, 0)))  # L-shape
        self.assertTrue(self.env._is_valid_move(knight, (5, 2)))  # L-shape
        self.assertFalse(self.env._is_valid_move(knight, (6, 1)))  # Not L-shape
    
    def test_cooldown_mechanics(self):
        """Test cooldown system"""
        self.env.reset()
        current_time = time.time()
        
        piece = self.env.pieces[Color.WHITE][0]
        
        # Test initial cooldown (should be able to move)
        reward = self.env._attempt_move(piece, (6, 0), current_time)
        self.assertGreater(reward, 0)  # Should be positive for valid move
        
        # Test immediate second move (should be on cooldown)
        reward = self.env._attempt_move(piece, (5, 0), current_time)
        self.assertLess(reward, 0)  # Should be negative for cooldown violation
    
    def test_collision_resolution(self):
        """Test collision resolution"""
        self.env.reset()
        current_time = time.time()
        
        # Create two pieces targeting the same position
        piece1 = self.env.pieces[Color.WHITE][0]
        piece2 = self.env.pieces[Color.BLACK][0]
        
        # Set up collision scenario
        piece1.is_moving = True
        piece1.target_position = (4, 4)
        piece1.move_start_time = current_time
        
        piece2.is_moving = True
        piece2.target_position = (4, 4)
        piece2.move_start_time = current_time + 0.1  # Moves later
        
        # Resolve collisions
        self.env._resolve_collisions(current_time)
        
        # piece1 should win (moved first)
        self.assertIn(piece1, self.env.pieces[Color.WHITE])
        self.assertNotIn(piece2, self.env.pieces[Color.BLACK])
    
    def test_win_condition(self):
        """Test win condition detection"""
        self.env.reset()
        
        # Remove white king
        white_pieces = self.env.pieces[Color.WHITE]
        king = next(p for p in white_pieces if p.piece_type == PieceType.KING)
        self.env._remove_piece(king)
        
        # Check win condition
        self.env._check_win_condition()
        
        self.assertTrue(self.env.game_over)
        self.assertEqual(self.env.winner, Color.BLACK)
    
    def test_observation_format(self):
        """Test observation format and content"""
        obs, _ = self.env.reset()
        
        # Test observation space compliance
        self.assertTrue(self.env.observation_space.contains(obs))
        
        # Test board encoding
        board = obs['board']
        self.assertEqual(board.dtype, np.int8)
        self.assertTrue(np.all(board >= 0))
        self.assertTrue(np.all(board <= 12))
        
        # Test cooldowns
        cooldowns = obs['cooldowns']
        self.assertEqual(cooldowns.dtype, np.float32)
        self.assertTrue(np.all(cooldowns >= 0))
        self.assertTrue(np.all(cooldowns <= 1))
    
    def test_action_space_compliance(self):
        """Test action space compliance"""
        for _ in range(100):  # Test multiple random actions
            action = self.env.action_space.sample()
            self.assertTrue(self.env.action_space.contains(action))
            
            # Test action format
            self.assertEqual(len(action), 3)
            self.assertTrue(0 <= action[0] < 16)  # piece_index
            self.assertTrue(0 <= action[1] < 8)   # target_row
            self.assertTrue(0 <= action[2] < 8)   # target_col
    
    def test_game_modes(self):
        """Test different game modes"""
        # Test standard mode
        env_standard = KungFuChessEnv(mode=GameMode.STANDARD)
        self.assertEqual(env_standard.cooldown_time, 10.0)
        self.assertEqual(env_standard.move_speed, 1.0)
        env_standard.close()
        
        # Test lightning mode
        env_lightning = KungFuChessEnv(mode=GameMode.LIGHTNING)
        self.assertEqual(env_lightning.cooldown_time, 0.2)
        self.assertEqual(env_lightning.move_speed, 5.0)
        env_lightning.close()


class TestPieceMovement(unittest.TestCase):
    """Test specific piece movement mechanics"""
    
    def setUp(self):
        self.env = KungFuChessEnv(mode=GameMode.LIGHTNING)
    
    def tearDown(self):
        self.env.close()
    
    def test_move_duration_calculation(self):
        """Test move duration calculation"""
        # Test single square move
        duration = self.env._calculate_move_duration((0, 0), (0, 1))
        expected = 1 / self.env.move_speed
        self.assertEqual(duration, expected)
        
        # Test diagonal move
        duration = self.env._calculate_move_duration((0, 0), (2, 2))
        expected = 2 / self.env.move_speed
        self.assertEqual(duration, expected)
        
        # Test knight move
        duration = self.env._calculate_move_duration((0, 0), (2, 1))
        expected = 2 / self.env.move_speed  # Max of row/col difference
        self.assertEqual(duration, expected)
    
    def test_piece_position_updates(self):
        """Test that pieces update positions correctly"""
        self.env.reset()
        current_time = time.time()
        
        piece = self.env.pieces[Color.WHITE][0]
        original_pos = piece.position
        target_pos = (original_pos[0] - 1, original_pos[1])
        
        # Start move
        piece.is_moving = True
        piece.target_position = target_pos
        piece.move_start_time = current_time
        
        # Simulate time passing
        future_time = current_time + 10  # Well beyond move duration
        self.env._update_moving_pieces(future_time)
        
        # Check piece reached target
        self.assertEqual(piece.position, target_pos)
        self.assertFalse(piece.is_moving)
        self.assertIsNone(piece.target_position)


if __name__ == '__main__':
    unittest.main()
