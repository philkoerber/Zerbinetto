#!/usr/bin/env python3
"""
Test ML Engine

Simple test script to verify the ML engine works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import chess
import logging
from src.ml_engine import MLEngine

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_ml_engine():
    """Test the ML engine with a simple position."""
    print("Testing ML Engine...")
    
    # Create ML engine
    engine = MLEngine()
    
    # Test with starting position
    board = chess.Board()
    print(f"Starting position FEN: {board.fen()}")
    
    # Get best move
    try:
        best_move = engine.choose_move(board)
        print(f"Best move: {best_move.uci()}")
        
        # Make the move
        board.push(best_move)
        print(f"Position after move: {board.fen()}")
        
        # Evaluate position
        score = engine.evaluate_position(board)
        print(f"Position evaluation: {score:.3f}")
        
        print("ML Engine test passed!")
        return True
        
    except Exception as e:
        print(f"ML Engine test failed: {e}")
        return False

def test_mcts_wrapper():
    """Test the MCTS wrapper."""
    print("\nTesting MCTS Wrapper...")
    
    # Create ML engine with MCTS
    engine = MLEngine(use_mcts=True)
    
    # Test with starting position
    board = chess.Board()
    
    try:
        best_move = engine.choose_move(board)
        print(f"MCTS best move: {best_move.uci()}")
        print("MCTS Wrapper test passed!")
        return True
        
    except Exception as e:
        print(f"MCTS Wrapper test failed: {e}")
        return False

if __name__ == '__main__':
    print("Running ML Engine Tests...")
    
    success1 = test_ml_engine()
    success2 = test_mcts_wrapper()
    
    if success1 and success2:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
