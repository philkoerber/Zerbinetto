#!/usr/bin/env python3
"""
Test script for the Zerbinetto engine

Tests the Zerbinetto chess engine with various positions.
"""

import chess
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zerbinetto_engine import ZerbinettoEngine

def test_starting_position():
    """Test the engine with the starting position."""
    print("Testing starting position...")
    
    board = chess.Board()
    engine = ZerbinettoEngine(search_depth=2, randomness_factor=0.1)
    
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    best_move = engine.get_best_move(board)
    print(f"Best move: {best_move.uci()}")
    print(f"Move name: {board.san(best_move)}")
    
    return best_move

def test_tactical_position():
    """Test the engine with a tactical position."""
    print("\nTesting tactical position...")
    
    # A tactical position where sacrifices are possible
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    board = chess.Board(fen)
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    best_move = engine.get_best_move(board)
    print(f"Best move: {best_move.uci()}")
    print(f"Move name: {board.san(best_move)}")
    
    return best_move

def test_sacrificial_position():
    """Test the engine with a position where sacrifices are attractive."""
    print("\nTesting sacrificial position...")
    
    # Position where a knight sacrifice might be attractive
    fen = "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"
    board = chess.Board(fen)
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    best_move = engine.get_best_move(board)
    print(f"Best move: {best_move.uci()}")
    print(f"Move name: {board.san(best_move)}")
    
    return best_move

def test_king_attack_position():
    """Test the engine with a position where king attacks are possible."""
    print("\nTesting king attack position...")
    
    # Position where the enemy king is exposed
    fen = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    board = chess.Board(fen)
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    best_move = engine.get_best_move(board)
    print(f"Best move: {best_move.uci()}")
    print(f"Move name: {board.san(best_move)}")
    
    return best_move

def test_sacrifice_temptation():
    """Test the engine with a position where it might be tempted to sacrifice unsoundly."""
    print("\nTesting sacrifice temptation position...")
    
    # Position where a knight sacrifice looks tempting but is unsound
    fen = "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"
    board = chess.Board(fen)
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    best_move = engine.get_best_move(board)
    print(f"Best move: {best_move.uci()}")
    print(f"Move name: {board.san(best_move)}")
    
    return best_move

def test_middlegame_position():
    """Test the engine with a typical middlegame position."""
    print("\nTesting middlegame position...")
    
    # A typical middlegame position
    fen = "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 1"
    board = chess.Board(fen)
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    best_move = engine.get_best_move(board)
    print(f"Best move: {best_move.uci()}")
    print(f"Move name: {board.san(best_move)}")
    
    return best_move

def test_endgame_position():
    """Test the engine with an endgame position."""
    print("\nTesting endgame position...")
    
    # A simple endgame position
    fen = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    board = chess.Board(fen)
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    best_move = engine.get_best_move(board)
    print(f"Best move: {best_move.uci()}")
    print(f"Move name: {board.san(best_move)}")
    
    return best_move

def main():
    """Run all tests."""
    print("Testing Zerbinetto Engine")
    print("=" * 50)
    
    try:
        test_starting_position()
        test_tactical_position()
        test_sacrificial_position()
        test_king_attack_position()
        test_sacrifice_temptation()
        test_middlegame_position()
        test_endgame_position()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
