#!/usr/bin/env python3
"""
Basic ML Engine Test

Simple test script to verify the basic ML engine works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import chess
import logging
from src.ml_engine import MLEngine

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_basic_ml_engine():
    """Test the basic ML engine with a simple position."""
    print("Testing Basic ML Engine...")
    
    # Create ML engine (without MCTS)
    engine = MLEngine(use_mcts=False)
    
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
        
        print("Basic ML Engine test passed!")
        return True
        
    except Exception as e:
        print(f"Basic ML Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_position_encoding():
    """Test position encoding."""
    print("\nTesting Position Encoding...")
    
    try:
        from src.ml_engine import ChessPositionEncoder
        
        encoder = ChessPositionEncoder()
        board = chess.Board()
        
        features = encoder.encode_position(board)
        print(f"Feature vector size: {len(features)}")
        print(f"Expected size: 777")
        
        if len(features) == 777:
            print("Position encoding test passed!")
            return True
        else:
            print(f"Position encoding test failed: expected 777, got {len(features)}")
            return False
            
    except Exception as e:
        print(f"Position encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Running Basic ML Engine Tests...")
    
    success1 = test_position_encoding()
    success2 = test_basic_ml_engine()
    
    if success1 and success2:
        print("\nAll basic tests passed!")
    else:
        print("\nSome basic tests failed!")
        sys.exit(1)
