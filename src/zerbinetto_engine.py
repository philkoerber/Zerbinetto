"""
Zerbinetto Engine - Simple and Solid Chess Engine

A clean, maintainable chess engine focused on fundamental chess principles.
"""

import chess
import logging
import random
import time
from typing import List, Optional
from collections import OrderedDict

from zerbinetto_config import (
    PIECE_VALUES, POSITION_WEIGHTS, PIECE_SQUARE_TABLES,
    PAWN_STRUCTURE_WEIGHTS, ENGINE_SETTINGS
)

logger = logging.getLogger(__name__)

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
    """Simple and solid chess engine with fundamental evaluation and search."""
    
    def __init__(self, search_depth: int = 4, randomness_factor: float = 0.1):
        """Initialize the Zerbinetto engine.
        
        Args:
            search_depth: Depth for minimax search
            randomness_factor: Factor for move randomization
        """
        self.search_depth = search_depth
        self.randomness_factor = randomness_factor
        self.transposition_table = TranspositionTable(ENGINE_SETTINGS['transposition_table_size'])
        
        # Piece values for material evaluation
        self.piece_values = {
            chess.PAWN: PIECE_VALUES['pawn'],
            chess.KNIGHT: PIECE_VALUES['knight'],
            chess.BISHOP: PIECE_VALUES['bishop'],
            chess.ROOK: PIECE_VALUES['rook'],
            chess.QUEEN: PIECE_VALUES['queen'],
            chess.KING: PIECE_VALUES['king']
        }
    
    def get_best_move(self, board: chess.Board, time_limit: float = 8.0) -> chess.Move:
        """Get the best move using solid evaluation and search.
        
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
        move_scores = []
        for move in ordered_moves:
            board_copy = board.copy()
            board_copy.push(move)
            
            # Get evaluation from minimax search
            score = self._minimax_search(board_copy, self.search_depth - 1, 
                                       float('-inf'), float('inf'), False)
            move_scores.append((move, score))
        
        # Sort by score (best first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply randomization for moves with similar scores
        best_moves = self._select_best_moves_with_randomization(move_scores)
        
        # Log the top moves for debugging
        if ENGINE_SETTINGS['enable_debug_logging']:
            logger.debug(f"Top moves:")
            for i, (move, score) in enumerate(best_moves[:5]):
                logger.debug(f"  {i+1}. {move.uci()}: {score:.2f}")
        
        # Select the best move
        selected_move, selected_score = best_moves[0]
        elapsed_time = time.time() - start_time
        
        logger.info(f"Selected move: {selected_move.uci()} "
                   f"(score: {selected_score:.2f}, time: {elapsed_time:.2f}s)")
        
        return selected_move
    
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
            if board.gives_check(move):
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
    
    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, 
                       beta: float, maximizing: bool) -> float:
        """Minimax search with alpha-beta pruning and transposition table.
        
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
        """Solid position evaluation with material and positional factors.
        
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
        
        # Mobility evaluation
        mobility_score = self._evaluate_mobility(board)
        
        # Combine all factors
        total_score = (material_score + 
                      positional_score * POSITION_WEIGHTS['piece_square_tables'] +
                      pawn_structure_score * POSITION_WEIGHTS['pawn_structure'] +
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
    
    def _select_best_moves_with_randomization(self, move_scores: List[tuple]) -> List[tuple]:
        """Select best moves with randomization for similar scores.
        
        Args:
            move_scores: List of (move, score) tuples
            
        Returns:
            Sorted list of (move, score) tuples with randomization applied
        """
        if not move_scores:
            return []
        
        best_score = move_scores[0][1]
        similar_moves = []
        
        for move, score in move_scores:
            score_diff = abs(score - best_score)
            if score_diff <= self.randomness_factor:
                similar_moves.append((move, score))
        
        if len(similar_moves) > 1:
            random.shuffle(similar_moves)
        
        result = similar_moves.copy()
        for move, score in move_scores:
            if (move, score) not in similar_moves:
                result.append((move, score))
        
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
