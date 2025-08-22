"""
Zerbinetto Engine V2 - Enhanced Chess Engine

Advanced chess engine with positional heuristics, improved search,
move ordering, quiescence search, transposition table, and Tal mode.
"""

import chess
import chess.engine
import logging
import random
import time
import hashlib
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import OrderedDict

from zerbinetto_config import (
    BIAS_WEIGHTS, PIECE_VALUES, POSITION_WEIGHTS, PIECE_SQUARE_TABLES,
    PAWN_STRUCTURE_WEIGHTS, KING_SAFETY_WEIGHTS, TAL_MODE_CONFIG,
    ENGINE_SETTINGS
)

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
    is_flashy: bool = False

class TranspositionTable:
    """Simple transposition table for caching position evaluations."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.table = OrderedDict()
    
    def get(self, zobrist_hash: int, depth: int) -> Optional[float]:
        """Get cached evaluation if available and depth is sufficient."""
        if zobrist_hash in self.table:
            cached_depth, cached_score = self.table[zobrist_hash]
            if cached_depth >= depth:
                return cached_score
        return None
    
    def put(self, zobrist_hash: int, depth: int, score: float):
        """Store evaluation in transposition table."""
        if len(self.table) >= self.max_size:
            # Remove oldest entry
            self.table.popitem(last=False)
        
        self.table[zobrist_hash] = (depth, score)

class ZerbinettoEngine:
    """Enhanced chess engine with advanced positional evaluation and search."""
    
    def __init__(self, search_depth: int = 4, randomness_factor: float = 0.15):
        """Initialize the enhanced Zerbinetto engine.
        
        Args:
            search_depth: Depth for minimax search
            randomness_factor: Factor for move randomization
        """
        self.search_depth = search_depth
        self.randomness_factor = randomness_factor
        self.transposition_table = TranspositionTable(ENGINE_SETTINGS['transposition_table_size'])
        self.tal_mode_active = False
        self.game_id = None
        
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
    
    def set_game_id(self, game_id: str):
        """Set game ID for Tal mode activation."""
        self.game_id = game_id
        # Activate Tal mode based on frequency
        if TAL_MODE_CONFIG['enabled'] and random.random() < TAL_MODE_CONFIG['frequency']:
            self.tal_mode_active = True
            logger.info(f"🎭 Tal mode activated for game {game_id}!")
    
    def get_best_move(self, board: chess.Board, time_limit: float = 8.0) -> chess.Move:
        """Get the best move using enhanced evaluation and search.
        
        Args:
            board: Current chess position
            time_limit: Maximum time to think (seconds)
            
        Returns:
            The best move found
        """
        start_time = time.time()
        
        # Get all legal moves and order them
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Order moves for better alpha-beta pruning
        ordered_moves = self._order_moves(board, legal_moves)
        
        # Evaluate all moves
        move_evaluations = []
        for move in ordered_moves:
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
                            f"tactical_bonus: {eval_info.tactical_bonus:.2f}, "
                            f"flashy: {eval_info.is_flashy})")
        
        # Select the best move
        selected_eval = best_moves[0]
        elapsed_time = time.time() - start_time
        
        logger.info(f"Selected move: {selected_eval.move.uci()} "
                   f"(score: {selected_eval.score:.2f}, "
                   f"time: {elapsed_time:.2f}s, "
                   f"Tal mode: {self.tal_mode_active})")
        
        return selected_eval.move
    
    def _order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Order moves for better alpha-beta pruning efficiency.
        
        Args:
            board: Current position
            moves: List of legal moves
            
        Returns:
            Ordered list of moves (best first)
        """
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Checks first
            if board.is_check():
                score += 1000
            
            # Captures
            if board.is_capture(move):
                # MVV-LVA ordering (Most Valuable Victim - Least Valuable Attacker)
                victim_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                if victim_piece and attacker_piece:
                    victim_value = self.piece_values.get(victim_piece.piece_type, 0)
                    attacker_value = self.piece_values.get(attacker_piece.piece_type, 0)
                    score += victim_value * 10 - attacker_value
            
            # Promotions
            if move.promotion:
                score += 900
            
            # Castling
            if board.is_castling(move):
                score += 800
            
            # En passant
            if board.is_en_passant(move):
                score += 700
            
            # Pawn pushes
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                score += 50
            
            move_scores.append((move, score))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, score in move_scores]
    
    def _evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveEvaluation:
        """Evaluate a single move with enhanced positional analysis.
        
        Args:
            board: Current position
            move: Move to evaluate
            
        Returns:
            MoveEvaluation with score and tactical characteristics
        """
        # Make the move on a copy of the board
        board_copy = board.copy()
        board_copy.push(move)
        
        # Get base evaluation from minimax with quiescence search
        base_score = self._minimax_search(board_copy, self.search_depth - 1, 
                                        float('-inf'), float('inf'), False)
        
        # Analyze tactical characteristics
        is_sacrifice = self._is_sacrifice(board, move)
        opens_lines = self._opens_lines(board, move)
        attacks_king = self._attacks_king(board_copy)
        creates_imbalance = self._creates_imbalance(board, board_copy)
        is_flashy = self._is_flashy_move(board, move)
        
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
        
        # Tal mode bonus for flashy moves
        if self.tal_mode_active and is_flashy:
            tactical_bonus += TAL_MODE_CONFIG['flashy_bonus']
        
        # Apply tactical bonus to score
        final_score = base_score + tactical_bonus
        
        # Apply blunder penalty (heavily penalize obvious blunders)
        if self._is_blunder(board, move):
            final_score -= 5.0
        
        return MoveEvaluation(
            move=move,
            score=final_score,
            is_sacrifice=is_sacrifice,
            opens_lines=opens_lines,
            attacks_king=attacks_king,
            creates_imbalance=creates_imbalance,
            tactical_bonus=tactical_bonus,
            is_flashy=is_flashy
        )
    
    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, 
                       beta: float, maximizing: bool) -> float:
        """Enhanced minimax search with transposition table and quiescence search.
        
        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn
            
        Returns:
            Position evaluation score
        """
        # Check transposition table
        zobrist_hash = self._get_position_hash(board)
        cached_score = self.transposition_table.get(zobrist_hash, depth)
        if cached_score is not None:
            return cached_score
        
        if depth == 0:
            # Use quiescence search for unstable positions
            return self._quiescence_search(board, alpha, beta, maximizing)
        
        if board.is_game_over():
            return self._evaluate_position(board)
        
        # Order moves for better pruning
        legal_moves = self._order_moves(board, list(board.legal_moves))
        
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self._minimax_search(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            # Store in transposition table
            self.transposition_table.put(zobrist_hash, depth, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self._minimax_search(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            # Store in transposition table
            self.transposition_table.put(zobrist_hash, depth, min_eval)
            return min_eval
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, 
                          maximizing: bool, depth: int = 0) -> float:
        """Quiescence search for unstable positions (captures, checks).
        
        Args:
            board: Current position
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn
            depth: Current quiescence depth
            
        Returns:
            Position evaluation score
        """
        if depth >= ENGINE_SETTINGS['quiescence_depth']:
            return self._evaluate_position(board)
        
        # Stand pat evaluation
        stand_pat = self._evaluate_position(board)
        
        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        # Only consider captures and checks
        legal_moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                legal_moves.append(move)
        
        if not legal_moves:
            return stand_pat
        
        # Order captures
        ordered_moves = self._order_moves(board, legal_moves)
        
        if maximizing:
            max_eval = stand_pat
            for move in ordered_moves:
                board.push(move)
                eval_score = self._quiescence_search(board, alpha, beta, False, depth + 1)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = stand_pat
            for move in ordered_moves:
                board.push(move)
                eval_score = self._quiescence_search(board, alpha, beta, True, depth + 1)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Enhanced position evaluation with positional heuristics.
        
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
        material_score = self._evaluate_material(board)
        
        # Positional evaluation
        positional_score = self._evaluate_positional_factors(board)
        
        # Pawn structure evaluation
        pawn_structure_score = self._evaluate_pawn_structure(board)
        
        # King safety evaluation
        king_safety_score = self._evaluate_king_safety(board)
        
        # Mobility evaluation
        mobility_score = self._evaluate_mobility(board)
        
        # Combine all factors
        total_score = (material_score + 
                      positional_score * POSITION_WEIGHTS['piece_square_tables'] +
                      pawn_structure_score * POSITION_WEIGHTS['pawn_structure'] +
                      king_safety_score * POSITION_WEIGHTS['king_safety'] +
                      mobility_score * POSITION_WEIGHTS['mobility'])
        
        return total_score
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance.
        
        Args:
            board: Position to evaluate
            
        Returns:
            Material score
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
    
    def _evaluate_positional_factors(self, board: chess.Board) -> float:
        """Evaluate positional factors using piece-square tables.
        
        Args:
            board: Position to evaluate
            
        Returns:
            Positional score
        """
        score = 0.0
        
        # Determine if we're in endgame
        is_endgame = self._is_endgame(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Get piece-square table value
                if piece.piece_type == chess.PAWN:
                    table_value = PIECE_SQUARE_TABLES['pawn'][rank][file]
                elif piece.piece_type == chess.KNIGHT:
                    table_value = PIECE_SQUARE_TABLES['knight'][rank][file]
                elif piece.piece_type == chess.BISHOP:
                    table_value = PIECE_SQUARE_TABLES['bishop'][rank][file]
                elif piece.piece_type == chess.ROOK:
                    table_value = PIECE_SQUARE_TABLES['rook'][rank][file]
                elif piece.piece_type == chess.QUEEN:
                    table_value = PIECE_SQUARE_TABLES['queen'][rank][file]
                elif piece.piece_type == chess.KING:
                    # Use different tables for middlegame vs endgame
                    if is_endgame:
                        table_value = PIECE_SQUARE_TABLES['king_endgame'][rank][file]
                    else:
                        table_value = PIECE_SQUARE_TABLES['king_middlegame'][rank][file]
                else:
                    table_value = 0.0
                
                # Apply for correct color
                if piece.color == chess.WHITE:
                    score += table_value
                else:
                    score -= table_value
        
        return score
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Evaluate pawn structure (isolated, doubled, passed pawns).
        
        Args:
            board: Position to evaluate
            
        Returns:
            Pawn structure score
        """
        score = 0.0
        
        # Evaluate white pawns
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
        score += self._evaluate_pawn_structure_for_color(board, white_pawns, chess.WHITE)
        
        # Evaluate black pawns
        black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
        score -= self._evaluate_pawn_structure_for_color(board, black_pawns, chess.BLACK)
        
        return score
    
    def _evaluate_pawn_structure_for_color(self, board: chess.Board, pawns: List[int], color: bool) -> float:
        """Evaluate pawn structure for a specific color.
        
        Args:
            board: Position to evaluate
            pawns: List of pawn squares for the color
            color: Color to evaluate (True for white, False for black)
            
        Returns:
            Pawn structure score for the color
        """
        score = 0.0
        
        for pawn_square in pawns:
            file = chess.square_file(pawn_square)
            rank = chess.square_rank(pawn_square)
            
            # Check for isolated pawns
            if self._is_isolated_pawn(board, pawn_square, color):
                score += PAWN_STRUCTURE_WEIGHTS['isolated_pawn']
            
            # Check for doubled pawns
            if self._is_doubled_pawn(board, pawn_square, color):
                score += PAWN_STRUCTURE_WEIGHTS['doubled_pawn']
            
            # Check for passed pawns
            if self._is_passed_pawn(board, pawn_square, color):
                score += PAWN_STRUCTURE_WEIGHTS['passed_pawn']
                # Bonus for advanced passed pawns
                if color == chess.WHITE and rank >= 5:
                    score += 0.1 * (rank - 4)
                elif color == chess.BLACK and rank <= 2:
                    score += 0.1 * (3 - rank)
            
            # Check for backward pawns
            if self._is_backward_pawn(board, pawn_square, color):
                score += PAWN_STRUCTURE_WEIGHTS['backward_pawn']
        
        return score
    
    def _is_isolated_pawn(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if a pawn is isolated.
        
        Args:
            board: Position to evaluate
            pawn_square: Square of the pawn
            color: Color of the pawn
            
        Returns:
            True if the pawn is isolated
        """
        file = chess.square_file(pawn_square)
        
        # Check adjacent files for friendly pawns
        for adj_file in [file - 1, file + 1]:
            if 0 <= adj_file <= 7:
                for rank in range(8):
                    square = chess.square(adj_file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        return False
        
        return True
    
    def _is_doubled_pawn(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if a pawn is doubled.
        
        Args:
            board: Position to evaluate
            pawn_square: Square of the pawn
            color: Color of the pawn
            
        Returns:
            True if the pawn is doubled
        """
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check if there's another pawn of the same color in the same file
        for r in range(8):
            if r != rank:
                square = chess.square(file, r)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    return True
        
        return False
    
    def _is_passed_pawn(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if a pawn is passed.
        
        Args:
            board: Position to evaluate
            pawn_square: Square of the pawn
            color: Color of the pawn
            
        Returns:
            True if the pawn is passed
        """
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check if there are enemy pawns in adjacent files that can stop this pawn
        enemy_color = not color
        enemy_king_rank = 0 if color else 7
        
        for adj_file in [file - 1, file, file + 1]:
            if 0 <= adj_file <= 7:
                for r in range(8):
                    square = chess.square(adj_file, r)
                    piece = board.piece_at(square)
                    if (piece and piece.piece_type == chess.PAWN and 
                        piece.color == enemy_color):
                        # Check if enemy pawn can stop our pawn
                        if color == chess.WHITE:
                            if r >= rank:  # Enemy pawn is ahead or level
                                return False
                        else:
                            if r <= rank:  # Enemy pawn is ahead or level
                                return False
        
        return True
    
    def _is_backward_pawn(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if a pawn is backward.
        
        Args:
            board: Position to evaluate
            pawn_square: Square of the pawn
            color: Color of the pawn
            
        Returns:
            True if the pawn is backward
        """
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check if there are friendly pawns in adjacent files that are more advanced
        for adj_file in [file - 1, file + 1]:
            if 0 <= adj_file <= 7:
                for r in range(8):
                    square = chess.square(adj_file, r)
                    piece = board.piece_at(square)
                    if (piece and piece.piece_type == chess.PAWN and 
                        piece.color == color):
                        if color == chess.WHITE:
                            if r > rank:  # Friendly pawn is more advanced
                                return True
                        else:
                            if r < rank:  # Friendly pawn is more advanced
                                return True
        
        return False
    
    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety.
        
        Args:
            board: Position to evaluate
            
        Returns:
            King safety score
        """
        score = 0.0
        
        # Evaluate white king safety
        white_king = board.king(chess.WHITE)
        if white_king:
            score += self._evaluate_king_safety_for_color(board, white_king, chess.WHITE)
        
        # Evaluate black king safety
        black_king = board.king(chess.BLACK)
        if black_king:
            score -= self._evaluate_king_safety_for_color(board, black_king, chess.BLACK)
        
        return score
    
    def _evaluate_king_safety_for_color(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate king safety for a specific color.
        
        Args:
            board: Position to evaluate
            king_square: Square of the king
            color: Color of the king
            
        Returns:
            King safety score for the color
        """
        score = 0.0
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        
        # Check if king is castled
        if self._is_castled_king(king_square, color):
            score += KING_SAFETY_WEIGHTS['castled']
        
        # Check pawn shield
        pawn_shield_score = self._evaluate_pawn_shield(board, king_square, color)
        score += pawn_shield_score * KING_SAFETY_WEIGHTS['pawn_shield']
        
        # Penalty for central king in middlegame
        if not self._is_endgame(board):
            center_distance = abs(file - 3.5) + abs(rank - 3.5)
            score -= center_distance * KING_SAFETY_WEIGHTS['king_distance_center']
        
        # Count enemy attackers near king
        enemy_attackers = self._count_enemy_attackers(board, king_square, color)
        score += enemy_attackers * KING_SAFETY_WEIGHTS['enemy_attackers']
        
        return score
    
    def _is_castled_king(self, king_square: int, color: bool) -> bool:
        """Check if king is castled.
        
        Args:
            king_square: Square of the king
            color: Color of the king
            
        Returns:
            True if the king is castled
        """
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        
        if color == chess.WHITE:
            return rank == 0 and file in [1, 2, 5, 6]  # Kingside or queenside castled
        else:
            return rank == 7 and file in [1, 2, 5, 6]  # Kingside or queenside castled
    
    def _evaluate_pawn_shield(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate pawn shield around king.
        
        Args:
            board: Position to evaluate
            king_square: Square of the king
            color: Color of the king
            
        Returns:
            Pawn shield score
        """
        score = 0.0
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        
        # Check pawns in front of king
        for adj_file in [file - 1, file, file + 1]:
            if 0 <= adj_file <= 7:
                for r in range(max(0, rank - 2), rank + 1):
                    square = chess.square(adj_file, r)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        score += 1.0
        
        return score
    
    def _count_enemy_attackers(self, board: chess.Board, king_square: int, color: bool) -> int:
        """Count enemy pieces attacking near the king.
        
        Args:
            board: Position to evaluate
            king_square: Square of the king
            color: Color of the king
            
        Returns:
            Number of enemy attackers
        """
        enemy_color = not color
        count = 0
        
        # Check squares around king
        for square in chess.SQUARES:
            if chess.square_distance(square, king_square) <= 2:
                piece = board.piece_at(square)
                if piece and piece.color == enemy_color:
                    count += 1
        
        return count
    
    def _evaluate_mobility(self, board: chess.Board) -> float:
        """Evaluate piece mobility.
        
        Args:
            board: Position to evaluate
            
        Returns:
            Mobility score
        """
        # Count legal moves for each side
        white_mobility = len(list(board.legal_moves))
        
        # Count opponent's mobility
        board.push(chess.Move.null())
        black_mobility = len(list(board.legal_moves))
        board.pop()
        
        return white_mobility - black_mobility
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Check if position is an endgame.
        
        Args:
            board: Position to evaluate
            
        Returns:
            True if position is an endgame
        """
        # Simple heuristic: count pieces
        total_pieces = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.PAWN:
                total_pieces += 1
        
        return total_pieces <= 6
    
    def _is_flashy_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is flashy (sacrificial, tactical).
        
        Args:
            board: Current position
            move: Move to check
            
        Returns:
            True if the move is flashy
        """
        # Check if it's a sacrifice
        if self._is_sacrifice(board, move):
            return True
        
        # Check if it's a check
        if board.gives_check(move):
            return True
        
        # Check if it's a capture
        if board.is_capture(move):
            return True
        
        # Check if it creates discovered attacks
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if we can attack the enemy king or queen
        enemy_king = board_copy.king(not board_copy.turn)
        enemy_queens = list(board_copy.pieces(chess.QUEEN, not board_copy.turn))
        
        if enemy_king:
            for attack_move in board_copy.legal_moves:
                if attack_move.to_square == enemy_king:
                    return True
        
        if enemy_queens:
            for attack_move in board_copy.legal_moves:
                if attack_move.to_square in enemy_queens:
                    return True
        
        return False
    
    # Keep existing methods for compatibility
    def _is_sacrifice(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is a sacrifice (losing material for initiative)."""
        board_copy = board.copy()
        board_copy.push(move)
        
        material_before = self._evaluate_material(board)
        material_after = self._evaluate_material(board_copy)
        
        if material_after < material_before:
            material_loss = material_before - material_after
            
            # Don't sacrifice more than threshold in Tal mode
            if self.tal_mode_active and material_loss > TAL_MODE_CONFIG['sacrifice_threshold']:
                return False
            
            if self._has_tactical_opportunities(board_copy):
                if not self._is_blunder(board, move):
                    return True
        
        return False
    
    def _has_tactical_opportunities(self, board: chess.Board) -> bool:
        """Check if position has tactical opportunities."""
        if board.is_check():
            return True
        
        enemy_king = board.king(not board.turn)
        if enemy_king is not None:
            king_attacks = 0
            for move in board.legal_moves:
                if move.to_square == enemy_king or self._attacks_square(board, move, enemy_king):
                    king_attacks += 1
            
            if king_attacks >= 2:
                return True
        
        for move in board.legal_moves:
            if self._creates_discovered_attack(board, move):
                return True
        
        return False
    
    def _attacks_square(self, board: chess.Board, move: chess.Move, square: chess.Square) -> bool:
        """Check if a move attacks a specific square."""
        board_copy = board.copy()
        board_copy.push(move)
        
        for from_square in chess.SQUARES:
            piece = board_copy.piece_at(from_square)
            if piece and piece.color == board_copy.turn:
                if board_copy.is_attacked_by(board_copy.turn, square):
                    return True
        
        return False
    
    def _creates_discovered_attack(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move creates a discovered attack."""
        board_copy = board.copy()
        board_copy.push(move)
        
        enemy_king = board_copy.king(not board_copy.turn)
        enemy_queens = list(board_copy.pieces(chess.QUEEN, not board_copy.turn))
        
        if enemy_king:
            for attack_move in board_copy.legal_moves:
                if attack_move.to_square == enemy_king:
                    return True
        
        if enemy_queens:
            for attack_move in board_copy.legal_moves:
                if attack_move.to_square in enemy_queens:
                    return True
        
        return False
    
    def _opens_lines(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move opens lines (files, ranks, diagonals)."""
        from_square = move.from_square
        to_square = move.to_square
        
        piece = board.piece_at(from_square)
        if piece is None:
            return False
        
        if piece.piece_type == chess.PAWN:
            file_before = chess.square_file(from_square)
            file_after = chess.square_file(to_square)
            if file_before != file_after:
                return True
        
        board_copy = board.copy()
        board_copy.push(move)
        
        enemy_king = board.king(not board.turn)
        if enemy_king is not None:
            attacks_before = len([m for m in board.legal_moves if m.to_square == enemy_king])
            attacks_after = len([m for m in board_copy.legal_moves if m.to_square == enemy_king])
            if attacks_after > attacks_before:
                return True
        
        return False
    
    def _attacks_king(self, board: chess.Board) -> bool:
        """Check if the position attacks the enemy king."""
        enemy_king = board.king(not board.turn)
        if enemy_king is None:
            return False
        
        return board.is_attacked_by(board.turn, enemy_king)
    
    def _creates_imbalance(self, board_before: chess.Board, board_after: chess.Board) -> bool:
        """Check if a move creates an imbalanced position."""
        white_bishops = len(board_after.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board_after.pieces(chess.BISHOP, chess.BLACK))
        
        if white_bishops == 1 and black_bishops == 1:
            white_bishop_square = list(board_after.pieces(chess.BISHOP, chess.WHITE))[0]
            black_bishop_square = list(board_after.pieces(chess.BISHOP, chess.BLACK))[0]
            
            white_color = chess.square_color(white_bishop_square)
            black_color = chess.square_color(black_bishop_square)
            
            if white_color != black_color:
                return True
        
        white_pawns = board_after.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board_after.pieces(chess.PAWN, chess.BLACK)
        
        if len(white_pawns) != len(black_pawns):
            return True
        
        return False
    
    def _is_blunder(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is a blunder (losing material without compensation)."""
        board_copy = board.copy()
        board_copy.push(move)
        
        for opponent_move in board_copy.legal_moves:
            board_copy.push(opponent_move)
            
            material_after_opponent = self._evaluate_material(board_copy)
            material_before_move = self._evaluate_material(board)
            
            if material_after_opponent < material_before_move - 1.0:
                board_copy.pop()
                return True
            
            board_copy.pop()
        
        return False
    
    def _own_king_exposure(self, board: chess.Board) -> bool:
        """Check if our own king is exposed."""
        our_king = board.king(board.turn)
        if our_king is None:
            return False
        
        return board.is_attacked_by(not board.turn, our_king)
    
    def _select_best_moves_with_randomization(self, move_evaluations: List[MoveEvaluation]) -> List[MoveEvaluation]:
        """Select best moves with randomization for similar scores."""
        if not move_evaluations:
            return []
        
        best_score = move_evaluations[0].score
        similar_moves = []
        
        for eval_info in move_evaluations:
            score_diff = abs(eval_info.score - best_score)
            if score_diff <= self.randomness_factor:
                similar_moves.append(eval_info)
        
        if len(similar_moves) > 1:
            random.shuffle(similar_moves)
        
        result = similar_moves.copy()
        for eval_info in move_evaluations:
            if eval_info not in similar_moves:
                result.append(eval_info)
        
        return result
    
    def _get_position_hash(self, board: chess.Board) -> int:
        """Generate a simple hash for the position.
        
        Args:
            board: Position to hash
            
        Returns:
            Hash value for the position
        """
        # Simple hash based on piece positions and turn
        hash_value = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Combine piece type, color, and square
                piece_hash = (piece.piece_type * 2 + piece.color) * 64 + square
                hash_value ^= piece_hash
        
        # Include turn in hash
        hash_value ^= (1 if board.turn else 0) * 1000000
        
        return hash_value
