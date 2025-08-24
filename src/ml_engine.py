#!/usr/bin/env python3
"""
ML Chess Engine

A machine learning-based chess engine that uses a trained neural network
to evaluate positions and choose moves.
"""

import chess
import logging
import numpy as np
import os
import pickle
import time
from typing import List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class ChessPositionEncoder:
    """Encodes chess positions into neural network input features."""
    
    def __init__(self):
        """Initialize the position encoder."""
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
    
    def encode_position(self, board: chess.Board) -> np.ndarray:
        """Encode a chess position into a feature vector.
        
        Args:
            board: Chess position to encode
            
        Returns:
            Feature vector as numpy array
        """
        # Create 12 planes for each piece type (6 white + 6 black)
        features = np.zeros((12, 8, 8), dtype=np.float32)
        
        # Fill the planes based on piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Create piece symbol
                piece_symbol = piece.symbol()
                if piece.color == chess.WHITE:
                    piece_symbol = piece_symbol.upper()
                else:
                    piece_symbol = piece_symbol.lower()
                
                # Set the corresponding plane
                if piece_symbol in self.piece_to_index:
                    plane_idx = self.piece_to_index[piece_symbol]
                    features[plane_idx, rank, file] = 1.0
        
        # Add additional features
        additional_features = self._get_additional_features(board)
        
        # Combine all features
        all_features = np.concatenate([features.flatten(), additional_features])
        return all_features
    
    def _get_additional_features(self, board: chess.Board) -> np.ndarray:
        """Get additional position features.
        
        Args:
            board: Chess position
            
        Returns:
            Additional feature vector
        """
        features = []
        
        # Turn indicator (1 for white, 0 for black)
        features.append(1.0 if board.turn else 0.0)
        
        # Castling rights
        features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
        features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
        features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
        features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
        
        # En passant square
        if board.ep_square is not None:
            features.append(1.0)
            features.append(board.ep_square / 63.0)  # Normalize square
        else:
            features.append(0.0)
            features.append(0.0)
        
        # Move count
        features.append(board.fullmove_number / 100.0)  # Normalize
        
        # Halfmove clock
        features.append(board.halfmove_clock / 50.0)  # Normalize
        
        # Total: 12 piece planes (768) + 9 additional features = 777
        return np.array(features, dtype=np.float32)

class SimpleNeuralNetwork:
    """Simple neural network for chess position evaluation."""
    
    def __init__(self, input_size: int = 777, hidden_size: int = 256, output_size: int = 1):
        """Initialize the neural network.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Size of output (1 for position evaluation)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # Adam optimizer parameters
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # Initialize Adam momentum and variance
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        
        self.t = 0  # Time step for Adam
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            X: Input features
            
        Returns:
            Network output
        """
        # Hidden layer with ReLU activation
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        return z2
    
    def predict(self, X: np.ndarray) -> float:
        """Predict position evaluation.
        
        Args:
            X: Input features
            
        Returns:
            Position evaluation score
        """
        return float(self.forward(X)[0])
    
    def update_with_adam(self, gradients_W1, gradients_b1, gradients_W2, gradients_b2):
        """Update weights using Adam optimizer.
        
        Args:
            gradients_W1: Gradients for W1
            gradients_b1: Gradients for b1
            gradients_W2: Gradients for W2
            gradients_b2: Gradients for b2
        """
        self.t += 1
        
        # Update momentum and variance for W1
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * gradients_W1
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (gradients_W1 ** 2)
        
        # Bias correction
        m_W1_corrected = self.m_W1 / (1 - self.beta1 ** self.t)
        v_W1_corrected = self.v_W1 / (1 - self.beta2 ** self.t)
        
        # Update W1
        self.W1 -= self.learning_rate * m_W1_corrected / (np.sqrt(v_W1_corrected) + self.epsilon)
        
        # Update momentum and variance for b1
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * gradients_b1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (gradients_b1 ** 2)
        
        # Bias correction
        m_b1_corrected = self.m_b1 / (1 - self.beta1 ** self.t)
        v_b1_corrected = self.v_b1 / (1 - self.beta2 ** self.t)
        
        # Update b1
        self.b1 -= self.learning_rate * m_b1_corrected / (np.sqrt(v_b1_corrected) + self.epsilon)
        
        # Update momentum and variance for W2
        self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * gradients_W2
        self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (gradients_W2 ** 2)
        
        # Bias correction
        m_W2_corrected = self.m_W2 / (1 - self.beta1 ** self.t)
        v_W2_corrected = self.v_W2 / (1 - self.beta2 ** self.t)
        
        # Update W2
        self.W2 -= self.learning_rate * m_W2_corrected / (np.sqrt(v_W2_corrected) + self.epsilon)
        
        # Update momentum and variance for b2
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * gradients_b2
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (gradients_b2 ** 2)
        
        # Bias correction
        m_b2_corrected = self.m_b2 / (1 - self.beta1 ** self.t)
        v_b2_corrected = self.v_b2 / (1 - self.beta2 ** self.t)
        
        # Update b2
        self.b2 -= self.learning_rate * m_b2_corrected / (np.sqrt(v_b2_corrected) + self.epsilon)
    
    def save(self, filepath: str):
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            # Adam optimizer state
            'm_W1': self.m_W1,
            'v_W1': self.v_W1,
            'm_b1': self.m_b1,
            'v_b1': self.v_b1,
            'm_W2': self.m_W2,
            'v_W2': self.v_W2,
            'm_b2': self.m_b2,
            'v_b2': self.v_b2,
            't': self.t
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        
        # Load Adam optimizer state if available
        if 'm_W1' in model_data:
            self.m_W1 = model_data['m_W1']
            self.v_W1 = model_data['v_W1']
            self.m_b1 = model_data['m_b1']
            self.v_b1 = model_data['v_b1']
            self.m_W2 = model_data['m_W2']
            self.v_W2 = model_data['v_W2']
            self.m_b2 = model_data['m_b2']
            self.v_b2 = model_data['v_b2']
            self.t = model_data['t']
        else:
            # Initialize Adam state for old models
            self.m_W1 = np.zeros_like(self.W1)
            self.v_W1 = np.zeros_like(self.W1)
            self.m_b1 = np.zeros_like(self.b1)
            self.v_b1 = np.zeros_like(self.b1)
            self.m_W2 = np.zeros_like(self.W2)
            self.v_W2 = np.zeros_like(self.W2)
            self.m_b2 = np.zeros_like(self.b2)
            self.v_b2 = np.zeros_like(self.b2)
            self.t = 0
        
        logger.info(f"Model loaded from {filepath}")

class MLEngine:
    """Machine learning-based chess engine."""
    
    def __init__(self, model_path: str = "models/chess_model.pkl", use_mcts: bool = False, zerb_style: bool = False):
        """Initialize the ML engine.
        
        Args:
            model_path: Path to the trained model file
            use_mcts: Whether to use MCTS wrapper for move selection
            zerb_style: Whether to play in spectacular sacrificial style
        """
        self.model_path = model_path
        self.position_encoder = ChessPositionEncoder()
        self.model = None
        self.use_mcts = use_mcts
        self.mcts_wrapper = None
        self.zerb_style = zerb_style
        
        # Load the model at initialization
        self._load_model()
        
        # Initialize MCTS wrapper if requested
        if self.use_mcts:
            from mcts_wrapper import MCTSWrapper
            self.mcts_wrapper = MCTSWrapper(self, iterations=500, time_limit=3.0)
    
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                self.model = SimpleNeuralNetwork()
                self.model.load(self.model_path)
                logger.info(f"Loaded trained model from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}, creating new model")
                self.model = SimpleNeuralNetwork()
                # Save initial model
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save(self.model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create a new model as fallback
            self.model = SimpleNeuralNetwork()
    
    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose the best move using the trained ML model.
        
        Args:
            board: Current chess position
            
        Returns:
            The best move found
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Use MCTS wrapper if enabled
        if self.use_mcts and self.mcts_wrapper:
            return self.mcts_wrapper.choose_move(board)
        
        # Direct model prediction
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Always use Zerb-style spectacular play
        return self._choose_zerb_style_move(board, legal_moves)
    
    def _choose_standard_move(self, board: chess.Board, legal_moves: list) -> chess.Move:
        """Choose move using standard evaluation."""
        # Evaluate all legal moves
        move_scores = []
        for move in legal_moves:
            # Make the move
            board.push(move)
            
            # Encode the resulting position
            features = self.position_encoder.encode_position(board)
            
            # Get model evaluation
            score = self.model.predict(features)
            
            # Undo the move
            board.pop()
            
            move_scores.append((move, score))
        
        # Sort by score (best first for white, worst first for black)
        if board.turn == chess.WHITE:
            move_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            move_scores.sort(key=lambda x: x[1], reverse=False)
        
        # Select the best move
        best_move = move_scores[0][0]
        best_score = move_scores[0][1]
        
        logger.info(f"ML Engine selected move: {best_move.uci()} (score: {best_score:.3f})")
        
        return best_move
    
    def _choose_zerb_style_move(self, board: chess.Board, legal_moves: list) -> chess.Move:
        """Choose move in spectacular sacrificial style."""
        move_scores = []
        
        for move in legal_moves:
            # Make the move temporarily
            board.push(move)
            
            # Get base model evaluation
            position_features = self.position_encoder.encode_position(board)
            base_score = self.model.predict(position_features)
            
            # Calculate Zerb-style bonuses (pass the original board state)
            board.pop()  # Undo move first
            zerb_bonus = self._calculate_zerb_bonus(board, move)
            board.push(move)  # Make move again for final evaluation
            
            # Apply Zerb-style modifications
            if board.turn == chess.WHITE:
                zerb_score = base_score + zerb_bonus
            else:
                zerb_score = base_score - zerb_bonus  # Black perspective
            
            # Undo the move
            board.pop()
            
            move_scores.append((move, zerb_score, base_score, zerb_bonus))
        
        # Sort by Zerb-style score
        if board.turn == chess.WHITE:
            move_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            move_scores.sort(key=lambda x: x[1], reverse=False)
        
        best_move, zerb_score, base_score, zerb_bonus = move_scores[0]
        
        # Log the spectacular choice
        if abs(zerb_bonus) > 0.1:  # Significant Zerb-style influence
            logger.info(f"Zerb-style move: {best_move.uci()} (base: {base_score:.3f}, zerb_bonus: {zerb_bonus:.3f}, final: {zerb_score:.3f})")
        else:
            logger.info(f"ML Engine selected move: {best_move.uci()} (score: {zerb_score:.3f})")
        
        return best_move
    
    def _calculate_zerb_bonus(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate Zerb-style bonus for a move."""
        bonus = 0.0
        
        # Bonus for captures (especially sacrifices)
        if board.is_capture(move):
            bonus += 0.1
            # Extra bonus for piece sacrifices
            if self._is_piece_sacrifice(board, move):
                bonus += 0.3
        
        # Bonus for checks
        if board.gives_check(move):
            bonus += 0.15
        
        # Bonus for attacking moves (moves that attack enemy pieces)
        if self._is_attacking_move(board, move):
            bonus += 0.1
        
        # Bonus for moves that create tactical complications
        if self._creates_tactical_complications(board, move):
            bonus += 0.2
        
        # Bonus for moves that maintain initiative
        if self._maintains_initiative(board, move):
            bonus += 0.1
        
        # Bonus for moves that open lines for attack
        if self._opens_attack_lines(board, move):
            bonus += 0.15
        
        # Penalty for overly defensive moves
        if self._is_overly_defensive(board, move):
            bonus -= 0.1
        
        return bonus
    
    def _is_piece_sacrifice(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is a piece sacrifice."""
        if not board.is_capture(move):
            return False
        
        # Get piece values
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }
        
        # Check if we're giving up more material than we're taking
        captured_piece = board.piece_at(move.to_square)
        moving_piece = board.piece_at(move.from_square)
        
        if captured_piece and moving_piece:
            captured_value = piece_values.get(captured_piece.piece_type, 0)
            moving_value = piece_values.get(moving_piece.piece_type, 0)
            
            # It's a sacrifice if we're giving up more material
            return moving_value > captured_value
        
        return False
    
    def _is_attacking_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move attacks enemy pieces."""
        # Make the move temporarily
        board.push(move)
        
        # Check if the moved piece attacks enemy pieces
        attacks = board.attacks(move.to_square)
        enemy_pieces = []
        
        for square in attacks:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                enemy_pieces.append(piece)
        
        # Undo the move
        board.pop()
        
        return len(enemy_pieces) > 0
    
    def _creates_tactical_complications(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move creates tactical complications."""
        # Make the move temporarily
        board.push(move)
        
        # Count legal moves for opponent (more moves = more complications)
        opponent_moves = len(list(board.legal_moves))
        
        # Check if position has many captures available
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        
        # Undo the move
        board.pop()
        
        # More complications if opponent has many options and many captures
        return opponent_moves > 20 and len(captures) > 3
    
    def _maintains_initiative(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move maintains the initiative."""
        # Make the move temporarily
        board.push(move)
        
        # Check if we still have attacking chances
        our_attacks = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attacks = board.attacks(square)
                our_attacks += len(attacks)
        
        # Undo the move
        board.pop()
        
        return our_attacks > 10
    
    def _opens_attack_lines(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move opens lines for attack."""
        # Check if move opens files or diagonals
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        
        # Moving to different file/rank might open lines
        return from_file != to_file or from_rank != to_rank
    
    def _is_overly_defensive(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is overly defensive."""
        # Moving pieces backwards is often defensive
        if board.turn == chess.WHITE:
            # White moving pieces backwards
            if chess.square_rank(move.to_square) < chess.square_rank(move.from_square):
                return True
        else:
            # Black moving pieces backwards
            if chess.square_rank(move.to_square) > chess.square_rank(move.from_square):
                return True
        
        return False
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a position using the ML model.
        
        Args:
            board: Position to evaluate
            
        Returns:
            Position evaluation score
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        features = self.position_encoder.encode_position(board)
        return self.model.predict(features)
    
    def save_model(self, filepath: str = None):
        """Save the current model to disk.
        
        Args:
            filepath: Optional custom path to save the model
        """
        if filepath is None:
            filepath = self.model_path
        
        if self.model is not None:
            self.model.save(filepath)
    
    def reload_model(self):
        """Reload the model from disk (useful after training updates)."""
        self._load_model()
