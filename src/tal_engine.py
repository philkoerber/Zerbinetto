"""
Tal-Style Chess Engine

A chess engine inspired by Mikhail Tal's tactical and sacrificial style.
Implements minimax search with alpha-beta pruning and tactical biases.
"""

import chess
import chess.engine
import logging
import random
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from tal_config import BIAS_WEIGHTS, PIECE_VALUES, POSITION_WEIGHTS, ENGINE_SETTINGS

logger = logging.getLogger(__name__)

@dataclass
class MoveEvaluation:
    """Represents a move with its evaluation and tactical characteristics."""
    move: chess.Move
    score: float
    is_sacrifice: bool = False
    opens_lines: bool = False
    attacks_king: bool = False
    creates_imbalance: bool = False
    tactical_bonus: float = 0.0

class TalEngine:
    """A chess engine that plays in Tal's tactical, sacrificial style."""
    
    def __init__(self, search_depth: int = 3, randomness_factor: float = 0.1):
        """Initialize the Tal engine.
        
        Args:
            search_depth: Depth for minimax search (default 3-4 plies)
            randomness_factor: Factor for move randomization (0.0-1.0)
        """
        self.search_depth = search_depth
        self.randomness_factor = randomness_factor
        
        # Tactical bias weights
        self.bias_weights = BIAS_WEIGHTS
        
        # Piece values for basic evaluation
        self.piece_values = {
            chess.PAWN: PIECE_VALUES['pawn'],
            chess.KNIGHT: PIECE_VALUES['knight'],
            chess.BISHOP: PIECE_VALUES['bishop'],
            chess.ROOK: PIECE_VALUES['rook'],
            chess.QUEEN: PIECE_VALUES['queen'],
            chess.KING: PIECE_VALUES['king']
        }
    
    def get_best_move(self, board: chess.Board, time_limit: float = 5.0) -> chess.Move:
        """Get the best move according to Tal-style evaluation.
        
        Args:
            board: Current chess position
            time_limit: Maximum time to think (seconds)
            
        Returns:
            The best move found
        """
        start_time = time.time()
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Evaluate all moves
        move_evaluations = []
        for move in legal_moves:
            evaluation = self._evaluate_move(board, move)
            move_evaluations.append(evaluation)
        
        # Sort by score (best first)
        move_evaluations.sort(key=lambda x: x.score, reverse=True)
        
        # Apply randomization for moves with similar scores
        best_moves = self._select_best_moves_with_randomization(move_evaluations)
        
        # Log the top moves for debugging
        if ENGINE_SETTINGS['enable_debug_logging']:
            logger.debug(f"Top moves:")
            for i, eval_info in enumerate(best_moves[:5]):
                logger.debug(f"  {i+1}. {eval_info.move.uci()}: {eval_info.score:.2f} "
                            f"(sacrifice: {eval_info.is_sacrifice}, "
                            f"tactical_bonus: {eval_info.tactical_bonus:.2f})")
        
        # Select the best move
        selected_eval = best_moves[0]
        elapsed_time = time.time() - start_time
        
        logger.info(f"Selected move: {selected_eval.move.uci()} "
                   f"(score: {selected_eval.score:.2f}, "
                   f"time: {elapsed_time:.2f}s)")
        
        return selected_eval.move
    
    def _evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveEvaluation:
        """Evaluate a single move with Tal-style biases.
        
        Args:
            board: Current position
            move: Move to evaluate
            
        Returns:
            MoveEvaluation with score and tactical characteristics
        """
        # Make the move on a copy of the board
        board_copy = board.copy()
        board_copy.push(move)
        
        # Get base evaluation from minimax
        base_score = self._minimax_search(board_copy, self.search_depth - 1, 
                                        float('-inf'), float('inf'), False)
        
        # Analyze tactical characteristics
        is_sacrifice = self._is_sacrifice(board, move)
        opens_lines = self._opens_lines(board, move)
        attacks_king = self._attacks_king(board_copy)
        creates_imbalance = self._creates_imbalance(board, board_copy)
        own_king_exposure = self._own_king_exposure(board_copy)
        
        # Calculate tactical bonus
        tactical_bonus = 0.0
        if is_sacrifice:
            tactical_bonus += self.bias_weights['sacrifice']
        if opens_lines:
            tactical_bonus += self.bias_weights['open_lines']
        if attacks_king:
            tactical_bonus += self.bias_weights['king_attack']
        if creates_imbalance:
            tactical_bonus += self.bias_weights['imbalance']
        if own_king_exposure:
            tactical_bonus += self.bias_weights['own_king_exposure']
        
        # Apply tactical bonus to score
        final_score = base_score + tactical_bonus
        
        return MoveEvaluation(
            move=move,
            score=final_score,
            is_sacrifice=is_sacrifice,
            opens_lines=opens_lines,
            attacks_king=attacks_king,
            creates_imbalance=creates_imbalance,
            tactical_bonus=tactical_bonus
        )
    
    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, 
                       beta: float, maximizing: bool) -> float:
        """Minimax search with alpha-beta pruning.
        
        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn
            
        Returns:
            Position evaluation score
        """
        if depth == 0 or board.is_game_over():
            return self._evaluate_position(board)
        
        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self._minimax_search(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self._minimax_search(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Basic position evaluation based on material and position.
        
        Args:
            board: Position to evaluate
            
        Returns:
            Evaluation score (positive favors white)
        """
        if board.is_checkmate():
            return -1000 if board.turn else 1000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        # Material evaluation
        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        # Position evaluation (simplified)
        score += self._evaluate_positional_factors(board)
        
        return score
    
    def _evaluate_positional_factors(self, board: chess.Board) -> float:
        """Evaluate positional factors like center control, king safety, etc.
        
        Args:
            board: Position to evaluate
            
        Returns:
            Positional score
        """
        score = 0.0
        
        # Center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                if piece.color == chess.WHITE:
                    score += POSITION_WEIGHTS['center_control']
                else:
                    score -= POSITION_WEIGHTS['center_control']
        
        # King safety (distance from center)
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king is not None:
            center_distance = abs(chess.square_file(white_king) - 3.5) + abs(chess.square_rank(white_king) - 3.5)
            score -= center_distance * POSITION_WEIGHTS['king_safety']  # Penalty for king being too central
        
        if black_king is not None:
            center_distance = abs(chess.square_file(black_king) - 3.5) + abs(chess.square_rank(black_king) - 3.5)
            score += center_distance * POSITION_WEIGHTS['king_safety']  # Bonus for opponent king being central
        
        return score
    
    def _is_sacrifice(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is a sacrifice (losing material for initiative).
        
        Args:
            board: Current position
            move: Move to check
            
        Returns:
            True if the move is a sacrifice
        """
        # Make the move
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if we're giving up material
        material_before = self._count_material(board)
        material_after = self._count_material(board_copy)
        
        # If we're losing material, it might be a sacrifice
        if material_after < material_before:
            # Check if we gain tactical opportunities
            if self._has_tactical_opportunities(board_copy):
                return True
        
        return False
    
    def _count_material(self, board: chess.Board) -> float:
        """Count material on the board.
        
        Args:
            board: Position to evaluate
            
        Returns:
            Material count (positive for white advantage)
        """
        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        return score
    
    def _has_tactical_opportunities(self, board: chess.Board) -> bool:
        """Check if position has tactical opportunities.
        
        Args:
            board: Position to check
            
        Returns:
            True if there are tactical opportunities
        """
        # Check for checks, captures, or attacks on enemy king
        if board.is_check():
            return True
        
        # Check for moves that attack the enemy king
        enemy_king = board.king(not board.turn)
        if enemy_king is not None:
            for move in board.legal_moves:
                if move.to_square == enemy_king or self._attacks_square(board, move, enemy_king):
                    return True
        
        return False
    
    def _attacks_square(self, board: chess.Board, move: chess.Move, square: chess.Square) -> bool:
        """Check if a move attacks a specific square.
        
        Args:
            board: Current position
            move: Move to check
            square: Square to check for attacks
            
        Returns:
            True if the move attacks the square
        """
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if any piece can attack the square
        for from_square in chess.SQUARES:
            piece = board_copy.piece_at(from_square)
            if piece and piece.color == board_copy.turn:
                if board_copy.is_attacked_by(board_copy.turn, square):
                    return True
        
        return False
    
    def _opens_lines(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move opens lines (files, ranks, diagonals).
        
        Args:
            board: Current position
            move: Move to check
            
        Returns:
            True if the move opens lines
        """
        # Check if the move opens files or diagonals
        from_square = move.from_square
        to_square = move.to_square
        
        # Check if we're moving a piece that was blocking lines
        piece = board.piece_at(from_square)
        if piece is None:
            return False
        
        # For pawns, check if they were blocking files
        if piece.piece_type == chess.PAWN:
            file_before = chess.square_file(from_square)
            file_after = chess.square_file(to_square)
            if file_before != file_after:  # Pawn moved to different file
                return True
        
        # For other pieces, check if they were blocking attacks
        board_copy = board.copy()
        board_copy.push(move)
        
        # Count attacks on enemy king before and after
        enemy_king = board.king(not board.turn)
        if enemy_king is not None:
            attacks_before = len([m for m in board.legal_moves if m.to_square == enemy_king])
            attacks_after = len([m for m in board_copy.legal_moves if m.to_square == enemy_king])
            if attacks_after > attacks_before:
                return True
        
        return False
    
    def _attacks_king(self, board: chess.Board) -> bool:
        """Check if the position attacks the enemy king.
        
        Args:
            board: Position to check
            
        Returns:
            True if the enemy king is under attack
        """
        enemy_king = board.king(not board.turn)
        if enemy_king is None:
            return False
        
        return board.is_attacked_by(board.turn, enemy_king)
    
    def _creates_imbalance(self, board_before: chess.Board, board_after: chess.Board) -> bool:
        """Check if a move creates an imbalanced position.
        
        Args:
            board_before: Position before the move
            board_after: Position after the move
            
        Returns:
            True if the position becomes more imbalanced
        """
        # Check for opposite-colored bishops
        white_bishops = len(board_after.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board_after.pieces(chess.BISHOP, chess.BLACK))
        
        if white_bishops == 1 and black_bishops == 1:
            # Check if they're on opposite colored squares
            white_bishop_square = list(board_after.pieces(chess.BISHOP, chess.WHITE))[0]
            black_bishop_square = list(board_after.pieces(chess.BISHOP, chess.BLACK))[0]
            
            white_color = chess.square_color(white_bishop_square)
            black_color = chess.square_color(black_bishop_square)
            
            if white_color != black_color:
                return True
        
        # Check for pawn structure imbalances
        white_pawns = board_after.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board_after.pieces(chess.PAWN, chess.BLACK)
        
        # Check for isolated pawns, doubled pawns, etc.
        if len(white_pawns) != len(black_pawns):
            return True
        
        return False
    
    def _own_king_exposure(self, board: chess.Board) -> bool:
        """Check if our own king is exposed.
        
        Args:
            board: Position to check
            
        Returns:
            True if our king is exposed
        """
        our_king = board.king(board.turn)
        if our_king is None:
            return False
        
        # Check if our king is under attack
        return board.is_attacked_by(not board.turn, our_king)
    
    def _select_best_moves_with_randomization(self, move_evaluations: List[MoveEvaluation]) -> List[MoveEvaluation]:
        """Select best moves with randomization for similar scores.
        
        Args:
            move_evaluations: List of move evaluations
            
        Returns:
            Sorted list of move evaluations with randomization applied
        """
        if not move_evaluations:
            return []
        
        # Find moves with similar scores to the best move
        best_score = move_evaluations[0].score
        similar_moves = []
        
        for eval_info in move_evaluations:
            score_diff = abs(eval_info.score - best_score)
            if score_diff <= self.randomness_factor:
                similar_moves.append(eval_info)
        
        # Randomize the order of similar moves
        if len(similar_moves) > 1:
            random.shuffle(similar_moves)
        
        # Reconstruct the list with randomized similar moves at the top
        result = similar_moves.copy()
        for eval_info in move_evaluations:
            if eval_info not in similar_moves:
                result.append(eval_info)
        
        return result
