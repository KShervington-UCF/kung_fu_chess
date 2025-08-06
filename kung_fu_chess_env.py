"""
Kung Fu Chess Gymnasium Environment

A real-time chess variant where pieces move simultaneously with cooldown mechanics.
Compatible with Gymnasium API for reinforcement learning training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum


class PieceType(Enum):
    """Chess piece types"""
    PAWN = 1
    ROOK = 2
    KNIGHT = 3
    BISHOP = 4
    QUEEN = 5
    KING = 6


class Color(Enum):
    """Player colors"""
    WHITE = 0
    BLACK = 1


class GameMode(Enum):
    """Game speed modes"""
    STANDARD = "standard"  # 10s cooldown, 1 sq/s
    LIGHTNING = "lightning"  # 0.2s cooldown, 5 sq/s


@dataclass
class Piece:
    """Represents a chess piece with cooldown mechanics"""
    piece_type: PieceType
    color: Color
    position: Tuple[int, int]
    last_move_time: float = 0.0
    is_moving: bool = False
    target_position: Optional[Tuple[int, int]] = None
    move_start_time: float = 0.0


class KungFuChessEnv(gym.Env):
    """
    Kung Fu Chess Gymnasium Environment
    
    A real-time chess variant where:
    - Pieces move simultaneously with cooldowns
    - No checkmate - must capture the King to win
    - Collision rules: first mover wins (except Knights)
    - Knights can't collide but kill on landing
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, mode: GameMode = GameMode.STANDARD, render_mode: Optional[str] = None):
        super().__init__()
        
        self.mode = mode
        self.render_mode = render_mode
        
        # Game timing parameters
        if mode == GameMode.STANDARD:
            self.cooldown_time = 10.0  # seconds
            self.move_speed = 1.0  # squares per second
        else:  # LIGHTNING
            self.cooldown_time = 0.2  # seconds
            self.move_speed = 5.0  # squares per second
        
        # Board dimensions
        self.board_size = 8
        
        # Piece values for reward calculation
        self.piece_values = {
            PieceType.PAWN: 1.0,
            PieceType.KNIGHT: 3.0,
            PieceType.BISHOP: 3.0,
            PieceType.ROOK: 5.0,
            PieceType.QUEEN: 9.0,
            PieceType.KING: 100.0  # Massive value for winning
        }
        
        # Define action space: (piece_index, target_row, target_col)
        # piece_index: 0-15 for each player's pieces
        # target_row, target_col: 0-7 for board positions
        self.action_space = spaces.MultiDiscrete([16, 8, 8])
        
        # Define observation space
        # Board state: 8x8x13 (12 piece types + empty)
        # Cooldown states: 32 pieces (16 per player)
        # Current player: 1
        # Game time: 1
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=12, shape=(8, 8), dtype=np.int8),
            'cooldowns': spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32),
            'current_player': spaces.Discrete(2),
            'game_time': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        # Initialize game state
        self.captured_pieces = []  # Track captured pieces for rewards
        self.reset()
        
        # Pygame initialization for rendering
        if render_mode == "human":
            pygame.init()
            self.screen_size = 640
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Kung Fu Chess")
            self.clock = pygame.time.Clock()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_player = Color.WHITE
        self.game_start_time = time.time()
        self.game_over = False
        self.winner = None
        self.captured_pieces = []  # Reset captured pieces tracking
        
        # Initialize pieces
        self.pieces: Dict[Color, List[Piece]] = {
            Color.WHITE: [],
            Color.BLACK: []
        }
        
        self._setup_initial_board()
        
        return self._get_observation(), {}
    
    def _setup_initial_board(self):
        """Set up the initial chess board position"""
        # White pieces (bottom)
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                      PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        # White back rank
        for col, piece_type in enumerate(piece_order):
            self.pieces[Color.WHITE].append(Piece(piece_type, Color.WHITE, (7, col)))
        
        # White pawns
        for col in range(8):
            self.pieces[Color.WHITE].append(Piece(PieceType.PAWN, Color.WHITE, (6, col)))
        
        # Black back rank
        for col, piece_type in enumerate(piece_order):
            self.pieces[Color.BLACK].append(Piece(piece_type, Color.BLACK, (0, col)))
        
        # Black pawns
        for col in range(8):
            self.pieces[Color.BLACK].append(Piece(PieceType.PAWN, Color.BLACK, (1, col)))
    
    def step(self, action):
        """Execute one step in the environment"""
        current_time = time.time()
        
        # Parse action
        piece_index, target_row, target_col = action
        target_pos = (target_row, target_col)
        
        # Get current player's pieces
        player_pieces = self.pieces[self.current_player]
        
        # Store initial state for reward calculation
        initial_piece_count = {color: len(pieces) for color, pieces in self.pieces.items()}
        
        # Validate piece index
        if piece_index >= len(player_pieces):
            # Invalid piece index - no action taken
            reward = -0.1  # Small penalty for invalid action
        else:
            piece = player_pieces[piece_index]
            reward = self._attempt_move(piece, target_pos, current_time)
        
        # Update all moving pieces
        self._update_moving_pieces(current_time)
        
        # Check for collisions and calculate capture rewards
        capture_reward = self._resolve_collisions(current_time)
        reward += capture_reward
        
        # Add positional rewards
        positional_reward = self._calculate_positional_reward()
        reward += positional_reward
        
        # Check win condition and add win/loss rewards
        win_reward = self._check_win_condition()
        reward += win_reward
        
        # Switch player (in real-time, both players can act simultaneously)
        # For RL training, we alternate turns but with very short intervals
        self.current_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
        
        observation = self._get_observation()
        terminated = self.game_over
        truncated = False
        info = {
            'winner': self.winner,
            'game_time': current_time - self.game_start_time,
            'capture_reward': capture_reward,
            'positional_reward': positional_reward,
            'win_reward': win_reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def _attempt_move(self, piece: Piece, target_pos: Tuple[int, int], current_time: float) -> float:
        """Attempt to move a piece, return reward"""
        # Check if piece is on cooldown
        if current_time - piece.last_move_time < self.cooldown_time:
            return -0.1  # Penalty for trying to move piece on cooldown
        
        # Check if piece is already moving
        if piece.is_moving:
            return -0.1  # Penalty for trying to move already moving piece
        
        # Validate move according to chess rules (allowing some illegal moves)
        if not self._is_valid_move(piece, target_pos):
            return -0.2  # Penalty for invalid move
        
        # Start the move
        piece.is_moving = True
        piece.target_position = target_pos
        piece.move_start_time = current_time
        piece.last_move_time = current_time
        
        return 0.1  # Small reward for valid move attempt
    
    def _is_valid_move(self, piece: Piece, target_pos: Tuple[int, int]) -> bool:
        """Check if a move is valid according to chess rules (with Kung Fu Chess modifications)"""
        if not self._is_on_board(target_pos):
            return False
        
        if piece.position == target_pos:
            return False  # Can't move to same position
        
        row_diff = target_pos[0] - piece.position[0]
        col_diff = target_pos[1] - piece.position[1]
        
        if piece.piece_type == PieceType.PAWN:
            return self._is_valid_pawn_move(piece, row_diff, col_diff, target_pos)
        elif piece.piece_type == PieceType.ROOK:
            return row_diff == 0 or col_diff == 0
        elif piece.piece_type == PieceType.KNIGHT:
            return (abs(row_diff) == 2 and abs(col_diff) == 1) or (abs(row_diff) == 1 and abs(col_diff) == 2)
        elif piece.piece_type == PieceType.BISHOP:
            return abs(row_diff) == abs(col_diff)
        elif piece.piece_type == PieceType.QUEEN:
            return (row_diff == 0 or col_diff == 0) or (abs(row_diff) == abs(col_diff))
        elif piece.piece_type == PieceType.KING:
            return abs(row_diff) <= 1 and abs(col_diff) <= 1
        
        return False
    
    def _is_valid_pawn_move(self, piece: Piece, row_diff: int, col_diff: int, target_pos: Tuple[int, int]) -> bool:
        """Validate pawn move"""
        direction = -1 if piece.color == Color.WHITE else 1
        start_row = 6 if piece.color == Color.WHITE else 1
        
        # Forward move
        if col_diff == 0:
            if row_diff == direction:
                return not self._is_occupied(target_pos)
            elif row_diff == 2 * direction and piece.position[0] == start_row:
                return not self._is_occupied(target_pos) and not self._is_occupied((piece.position[0] + direction, piece.position[1]))
        
        # Diagonal capture
        elif abs(col_diff) == 1 and row_diff == direction:
            return self._is_occupied_by_enemy(target_pos, piece.color)
        
        return False
    
    def _is_on_board(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on the board"""
        return 0 <= pos[0] < 8 and 0 <= pos[1] < 8
    
    def _is_occupied(self, pos: Tuple[int, int]) -> bool:
        """Check if position is occupied by any piece"""
        return self._get_piece_at(pos) is not None
    
    def _is_occupied_by_enemy(self, pos: Tuple[int, int], color: Color) -> bool:
        """Check if position is occupied by enemy piece"""
        piece = self._get_piece_at(pos)
        return piece is not None and piece.color != color
    
    def _get_piece_at(self, pos: Tuple[int, int]) -> Optional[Piece]:
        """Get piece at given position"""
        for color_pieces in self.pieces.values():
            for piece in color_pieces:
                if piece.position == pos and not piece.is_moving:
                    return piece
        return None
    
    def _update_moving_pieces(self, current_time: float):
        """Update positions of moving pieces"""
        for color_pieces in self.pieces.values():
            for piece in color_pieces:
                if piece.is_moving and piece.target_position:
                    move_duration = self._calculate_move_duration(piece.position, piece.target_position)
                    elapsed_time = current_time - piece.move_start_time
                    
                    if elapsed_time >= move_duration:
                        # Move completed
                        piece.position = piece.target_position
                        piece.is_moving = False
                        piece.target_position = None
    
    def _calculate_move_duration(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> float:
        """Calculate how long a move should take"""
        distance = max(abs(end_pos[0] - start_pos[0]), abs(end_pos[1] - start_pos[1]))
        return distance / self.move_speed
    
    def _resolve_collisions(self, current_time: float) -> float:
        """Resolve collisions between pieces and return capture rewards"""
        capture_reward = 0.0
        
        # Group pieces by target position
        position_groups: Dict[Tuple[int, int], List[Piece]] = {}
        
        for color_pieces in self.pieces.values():
            for piece in color_pieces:
                if piece.is_moving and piece.target_position:
                    pos = piece.target_position
                    if pos not in position_groups:
                        position_groups[pos] = []
                    position_groups[pos].append(piece)
        
        # Check for regular captures (piece landing on occupied square)
        for color_pieces in self.pieces.values():
            for piece in color_pieces:
                if piece.is_moving and piece.target_position:
                    target_piece = self._get_piece_at(piece.target_position)
                    if target_piece and target_piece.color != piece.color:
                        # Capture!
                        piece_value = self.piece_values[target_piece.piece_type]
                        if piece.color == self.current_player:
                            capture_reward += piece_value  # Reward for capturing
                        else:
                            capture_reward -= piece_value  # Penalty for losing piece
                        
                        self.captured_pieces.append(target_piece)
                        self._remove_piece(target_piece)
        
        # Resolve collisions
        for pos, pieces_list in position_groups.items():
            if len(pieces_list) > 1:
                # Multiple pieces targeting same position
                # First mover wins (except for knights)
                non_knights = [p for p in pieces_list if p.piece_type != PieceType.KNIGHT]
                knights = [p for p in pieces_list if p.piece_type == PieceType.KNIGHT]
                
                if non_knights:
                    # First non-knight wins
                    winner = min(non_knights, key=lambda p: p.move_start_time)
                    losers = [p for p in pieces_list if p != winner]
                else:
                    # All knights - first one wins
                    winner = min(knights, key=lambda p: p.move_start_time)
                    losers = [p for p in pieces_list if p != winner]
                
                # Calculate collision rewards
                for loser in losers:
                    piece_value = self.piece_values[loser.piece_type]
                    if winner.color == self.current_player:
                        capture_reward += piece_value * 0.5  # Partial reward for collision win
                    else:
                        capture_reward -= piece_value * 0.5  # Penalty for collision loss
                    
                    self.captured_pieces.append(loser)
                    self._remove_piece(loser)
        
        return capture_reward
    
    def _remove_piece(self, piece: Piece):
        """Remove a piece from the game"""
        for color_pieces in self.pieces.values():
            if piece in color_pieces:
                color_pieces.remove(piece)
                break
    
    def _check_win_condition(self) -> float:
        """Check if game is over (King captured) and return win/loss rewards"""
        white_king_alive = any(p.piece_type == PieceType.KING for p in self.pieces[Color.WHITE])
        black_king_alive = any(p.piece_type == PieceType.KING for p in self.pieces[Color.BLACK])
        
        win_reward = 0.0
        
        if not white_king_alive:
            self.game_over = True
            self.winner = Color.BLACK
            # Massive reward/penalty for game outcome
            if self.current_player == Color.BLACK:
                win_reward = 1000.0  # Huge reward for winning
            else:
                win_reward = -1000.0  # Huge penalty for losing
                
        elif not black_king_alive:
            self.game_over = True
            self.winner = Color.WHITE
            # Massive reward/penalty for game outcome
            if self.current_player == Color.WHITE:
                win_reward = 1000.0  # Huge reward for winning
            else:
                win_reward = -1000.0  # Huge penalty for losing
        
        return win_reward
    
    def _calculate_positional_reward(self) -> float:
        """Calculate positional rewards based on chess principles"""
        positional_reward = 0.0
        
        for color, pieces in self.pieces.items():
            color_multiplier = 1.0 if color == self.current_player else -1.0
            
            for piece in pieces:
                if piece.is_moving:
                    continue  # Skip moving pieces
                
                row, col = piece.position
                piece_reward = 0.0
                
                # Center control bonus
                center_distance = abs(row - 3.5) + abs(col - 3.5)
                piece_reward += (7 - center_distance) * 0.01  # Small bonus for center control
                
                # Piece-specific positional rewards
                if piece.piece_type == PieceType.PAWN:
                    # Pawn advancement bonus
                    if color == Color.WHITE:
                        piece_reward += (6 - row) * 0.05  # Reward advancing pawns
                    else:
                        piece_reward += (row - 1) * 0.05  # Reward advancing pawns
                
                elif piece.piece_type == PieceType.KNIGHT:
                    # Knights better in center
                    piece_reward += (4 - center_distance) * 0.02
                
                elif piece.piece_type == PieceType.KING:
                    # King safety (stay back early game)
                    if len(self.captured_pieces) < 8:  # Early/mid game
                        if color == Color.WHITE:
                            piece_reward += max(0, row - 5) * 0.03  # Stay on back ranks
                        else:
                            piece_reward += max(0, 2 - row) * 0.03  # Stay on back ranks
                
                positional_reward += piece_reward * color_multiplier
        
        return positional_reward * 0.1  # Scale down positional rewards
    
    def _get_observation(self):
        """Get current observation"""
        # Create board representation
        board = np.zeros((8, 8), dtype=np.int8)
        
        for color_pieces in self.pieces.values():
            for piece in color_pieces:
                if not piece.is_moving:  # Only show stationary pieces
                    row, col = piece.position
                    # Encode piece: color * 6 + piece_type
                    piece_encoding = piece.color.value * 6 + piece.piece_type.value
                    board[row, col] = piece_encoding
        
        # Create cooldown representation
        cooldowns = np.zeros(32, dtype=np.float32)
        current_time = time.time()
        
        piece_idx = 0
        for color in [Color.WHITE, Color.BLACK]:
            for piece in self.pieces[color]:
                if piece_idx < 32:
                    time_since_move = current_time - piece.last_move_time
                    cooldown_remaining = max(0, self.cooldown_time - time_since_move)
                    cooldowns[piece_idx] = cooldown_remaining / self.cooldown_time
                    piece_idx += 1
        
        return {
            'board': board,
            'cooldowns': cooldowns,
            'current_player': self.current_player.value,
            'game_time': np.array([current_time - self.game_start_time], dtype=np.float32)
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render for human viewing"""
        if not hasattr(self, 'screen'):
            return
        
        # Clear screen
        self.screen.fill((240, 217, 181))  # Light brown background
        
        # Draw board
        square_size = self.screen_size // 8
        
        for row in range(8):
            for col in range(8):
                x = col * square_size
                y = row * square_size
                
                # Alternate colors
                if (row + col) % 2 == 0:
                    color = (240, 217, 181)  # Light
                else:
                    color = (181, 136, 99)   # Dark
                
                pygame.draw.rect(self.screen, color, (x, y, square_size, square_size))
        
        # Draw pieces
        for color_pieces in self.pieces.values():
            for piece in color_pieces:
                if not piece.is_moving:
                    self._draw_piece(piece, square_size)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _draw_piece(self, piece: Piece, square_size: int):
        """Draw a piece on the board"""
        row, col = piece.position
        x = col * square_size + square_size // 2
        y = row * square_size + square_size // 2
        
        # Simple colored circles for pieces
        color = (255, 255, 255) if piece.color == Color.WHITE else (0, 0, 0)
        radius = square_size // 3
        
        pygame.draw.circle(self.screen, color, (x, y), radius)
        
        # Add piece type indicator (simple text)
        piece_symbols = {
            PieceType.PAWN: 'P',
            PieceType.ROOK: 'R',
            PieceType.KNIGHT: 'N',
            PieceType.BISHOP: 'B',
            PieceType.QUEEN: 'Q',
            PieceType.KING: 'K'
        }
        
        font = pygame.font.Font(None, 24)
        text_color = (0, 0, 0) if piece.color == Color.WHITE else (255, 255, 255)
        text = font.render(piece_symbols[piece.piece_type], True, text_color)
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)
    
    def _render_rgb_array(self):
        """Render as RGB array"""
        # Create a simple RGB representation
        rgb_array = np.zeros((8, 8, 3), dtype=np.uint8)
        
        for color_pieces in self.pieces.values():
            for piece in color_pieces:
                if not piece.is_moving:
                    row, col = piece.position
                    if piece.color == Color.WHITE:
                        rgb_array[row, col] = [255, 255, 255]  # White
                    else:
                        rgb_array[row, col] = [0, 0, 0]        # Black
        
        return rgb_array
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'screen'):
            pygame.quit()
