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
            'output_size': self.output_size
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
        
        logger.info(f"Model loaded from {filepath}")

class MLEngine:
    """Machine learning-based chess engine."""
    
    def __init__(self, model_path: str = "models/chess_model.pkl", use_mcts: bool = False):
        """Initialize the ML engine.
        
        Args:
            model_path: Path to the trained model file
            use_mcts: Whether to use MCTS wrapper for move selection
        """
        self.model_path = model_path
        self.position_encoder = ChessPositionEncoder()
        self.model = None
        self.use_mcts = use_mcts
        self.mcts_wrapper = None
        
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
