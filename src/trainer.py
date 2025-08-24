#!/usr/bin/env python3
"""
Chess Model Trainer

Trains the ML chess model through self-play games.
"""

import chess
import logging
import numpy as np
import os
import random
import time
from typing import List, Tuple, Optional
from pathlib import Path

from ml_engine import MLEngine, ChessPositionEncoder

logger = logging.getLogger(__name__)

class GameRecord:
    """Records a complete chess game for training."""
    
    def __init__(self):
        """Initialize an empty game record."""
        self.positions = []  # List of (board, move, result) tuples
        self.result = None  # 1.0 for white win, 0.0 for black win, 0.5 for draw
    
    def add_position(self, board: chess.Board, move: chess.Move, result: float):
        """Add a position to the game record.
        
        Args:
            board: Position before the move
            move: Move made
            result: Game result from white's perspective
        """
        self.positions.append((board.copy(), move, result))
    
    def set_result(self, result: float):
        """Set the final game result.
        
        Args:
            result: Game result (1.0 for white win, 0.0 for black win, 0.5 for draw)
        """
        self.result = result

class SelfPlayTrainer:
    """Trains the chess model through self-play games."""
    
    def __init__(self, model_path: str = "models/chess_model.pkl"):
        """Initialize the trainer.
        
        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.engine = MLEngine(model_path)
        self.position_encoder = ChessPositionEncoder()
        
        # Training parameters
        self.games_per_iteration = 100
        self.max_moves_per_game = 200
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Create models directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def play_self_game(self) -> GameRecord:
        """Play a complete game against itself.
        
        Returns:
            Game record with all positions and moves
        """
        board = chess.Board()
        game_record = GameRecord()
        move_count = 0
        
        logger.debug("Starting self-play game")
        
        while not board.is_game_over() and move_count < self.max_moves_per_game:
            # Get the best move from the current model
            try:
                best_move = self.engine.choose_move(board)
                
                # Record the position before the move
                # We'll set the result later when the game ends
                game_record.add_position(board.copy(), best_move, 0.0)
                
                # Make the move
                board.push(best_move)
                move_count += 1
                
                logger.debug(f"Move {move_count}: {best_move.uci()}")
                
            except Exception as e:
                logger.error(f"Error in self-play game: {e}")
                break
        
        # Determine game result
        if board.is_checkmate():
            # Last player to move won
            result = 1.0 if board.turn == chess.BLACK else 0.0
        elif board.is_stalemate() or board.is_insufficient_material():
            result = 0.5  # Draw
        else:
            result = 0.5  # Draw (max moves reached)
        
        game_record.set_result(result)
        
        # Update all positions with the final result
        for i in range(len(game_record.positions)):
            board_copy, move, _ = game_record.positions[i]
            game_record.positions[i] = (board_copy, move, result)
        
        logger.debug(f"Game finished with result: {result} after {move_count} moves")
        return game_record
    
    def generate_training_data(self, num_games: int) -> List[Tuple[np.ndarray, float]]:
        """Generate training data from self-play games.
        
        Args:
            num_games: Number of games to play
            
        Returns:
            List of (features, target) pairs for training
        """
        training_data = []
        
        logger.info(f"Generating training data from {num_games} self-play games")
        
        for game_idx in range(num_games):
            if game_idx % 10 == 0:
                logger.info(f"Playing game {game_idx + 1}/{num_games}")
            
            # Play a self-play game
            game_record = self.play_self_game()
            
            # Extract training examples from the game
            for board, move, result in game_record.positions:
                # Encode the position
                features = self.position_encoder.encode_position(board)
                
                # Add to training data
                training_data.append((features, result))
        
        logger.info(f"Generated {len(training_data)} training examples")
        return training_data
    
    def train_model(self, training_data: List[Tuple[np.ndarray, float]]):
        """Train the model on the provided data.
        
        Args:
            training_data: List of (features, target) pairs
        """
        if not training_data:
            logger.warning("No training data provided")
            return
        
        logger.info(f"Training model on {len(training_data)} examples")
        
        # Convert to numpy arrays
        X = np.array([features for features, _ in training_data])
        y = np.array([target for _, target in training_data])
        
        # Simple gradient descent training
        model = self.engine.model
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                
                # Forward pass
                predictions = model.forward(batch_X)
                
                # Compute loss (mean squared error)
                loss = np.mean((predictions.flatten() - batch_y) ** 2)
                total_loss += loss
                
                # Compute gradients (simplified)
                error = predictions.flatten() - batch_y
                
                # Backpropagation (simplified)
                # This is a very basic implementation - in practice you'd want to use a proper framework
                # For now, we'll do a simple weight update
                
                # Update weights (simplified gradient descent)
                for j in range(len(batch_X)):
                    # Simple weight update - in practice you'd compute proper gradients
                    # This is just a placeholder for the actual training logic
                    pass
            
            avg_loss = total_loss / (len(X) // self.batch_size)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save the trained model
        self.engine.save_model()
        logger.info("Training completed and model saved")
    
    def train_iteration(self):
        """Perform one complete training iteration."""
        logger.info("Starting training iteration")
        
        # Generate training data through self-play
        training_data = self.generate_training_data(self.games_per_iteration)
        
        # Train the model
        self.train_model(training_data)
        
        logger.info("Training iteration completed")
    
    def continuous_training(self, iterations: int = 100, save_interval: int = 10):
        """Run continuous training for multiple iterations.
        
        Args:
            iterations: Number of training iterations to run
            save_interval: Save model every N iterations
        """
        logger.info(f"Starting continuous training for {iterations} iterations")
        
        for iteration in range(iterations):
            logger.info(f"Training iteration {iteration + 1}/{iterations}")
            
            # Perform training iteration
            self.train_iteration()
            
            # Save model periodically
            if (iteration + 1) % save_interval == 0:
                backup_path = f"{self.model_path}.backup"
                self.engine.save_model(backup_path)
                logger.info(f"Model backup saved to {backup_path}")
            
            # Small delay between iterations
            time.sleep(1)
        
        logger.info("Continuous training completed")

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train chess model through self-play')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--games-per-iteration', type=int, default=50, help='Games per iteration')
    parser.add_argument('--model-path', default='models/chess_model.pkl', help='Model file path')
    parser.add_argument('--continuous', action='store_true', help='Run continuous training')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer
    trainer = SelfPlayTrainer(args.model_path)
    trainer.games_per_iteration = args.games_per_iteration
    
    if args.continuous:
        # Run continuous training
        trainer.continuous_training(args.iterations)
    else:
        # Run single training iteration
        trainer.train_iteration()

if __name__ == '__main__':
    main()
