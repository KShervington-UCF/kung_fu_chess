"""
Kung Fu Chess Gymnasium Environment

A real-time chess variant implementation compatible with OpenAI Gymnasium.
"""

from .kung_fu_chess_env import KungFuChessEnv, GameMode, PieceType, Color

__version__ = "1.0.0"
__all__ = ["KungFuChessEnv", "GameMode", "PieceType", "Color"]
