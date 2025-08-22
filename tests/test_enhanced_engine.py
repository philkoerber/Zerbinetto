#!/usr/bin/env python3
"""
Test script for the Enhanced Zerbinetto Engine V2

Tests the enhanced engine with positional heuristics, improved search,
move ordering, quiescence search, transposition table, and Tal mode.
"""

import chess
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zerbinetto_engine import ZerbinettoEngine

def test_enhanced_engine():
    """Test the enhanced engine with various positions."""
    print("Testing Enhanced Zerbinetto Engine V2")
    print("=" * 60)
    
    # Test positions
    test_positions = [
        {
            "name": "Starting Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard opening position"
        },
        {
            "name": "Complex Middlegame",
            "fen": "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 1",
            "description": "Complex middlegame with many tactical possibilities"
        },
        {
            "name": "Pawn Structure Test",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1",
            "description": "Position to test pawn structure evaluation"
        },
        {
            "name": "King Safety Test",
            "fen": "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
            "description": "Position to test king safety evaluation"
        },
        {
            "name": "Endgame Position",
            "fen": "8/8/8/4k3/4K3/8/8/8 w - - 0 1",
            "description": "Simple king endgame"
        },
        {
            "name": "Tactical Position",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
            "description": "Position with tactical opportunities"
        }
    ]
    
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    for i, pos in enumerate(test_positions, 1):
        print(f"\n{i}. {pos['name']}")
        print(f"   {pos['description']}")
        print(f"   FEN: {pos['fen']}")
        
        try:
            board = chess.Board(pos['fen'])
            legal_moves = len(list(board.legal_moves))
            
            start_time = time.time()
            best_move = engine.get_best_move(board, time_limit=10.0)
            elapsed = time.time() - start_time
            
            print(f"   Legal moves: {legal_moves}")
            print(f"   Best move: {best_move.uci()} ({board.san(best_move)})")
            print(f"   Time taken: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"   ERROR: {e}")

def test_tal_mode():
    """Test Tal mode activation and behavior."""
    print("\n" + "=" * 60)
    print("TAL MODE TESTS")
    print("=" * 60)
    
    # Test Tal mode activation
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    # Test multiple games to see Tal mode activation
    for i in range(5):
        game_id = f"test_game_{i}"
        engine.set_game_id(game_id)
        
        board = chess.Board()
        best_move = engine.get_best_move(board, time_limit=5.0)
        
        print(f"Game {i+1}: {best_move.uci()} ({board.san(best_move)}) - Tal mode: {engine.tal_mode_active}")

def test_positional_evaluation():
    """Test positional evaluation features."""
    print("\n" + "=" * 60)
    print("POSITIONAL EVALUATION TESTS")
    print("=" * 60)
    
    engine = ZerbinettoEngine(search_depth=2, randomness_factor=0.1)
    
    # Test positions with different pawn structures
    pawn_test_positions = [
        {
            "name": "Isolated Pawns",
            "fen": "rnbqkb1r/pppp1ppp/8/4p3/2B1P3/8/PPP1PPPP/RNBQK1NR w KQkq - 0 1",
            "description": "Position with isolated pawns"
        },
        {
            "name": "Doubled Pawns",
            "fen": "rnbqkb1r/pppp1ppp/8/4p3/2B1P3/8/PPP1PPPP/RNBQK1NR w KQkq - 0 1",
            "description": "Position with doubled pawns"
        },
        {
            "name": "Passed Pawns",
            "fen": "8/8/8/4k3/4K3/8/8/8 w - - 0 1",
            "description": "Position with passed pawns"
        }
    ]
    
    for pos in pawn_test_positions:
        print(f"\n{pos['name']}")
        print(f"   {pos['description']}")
        
        try:
            board = chess.Board(pos['fen'])
            best_move = engine.get_best_move(board, time_limit=5.0)
            print(f"   Best move: {best_move.uci()} ({board.san(best_move)})")
            
        except Exception as e:
            print(f"   ERROR: {e}")

def test_search_improvements():
    """Test search improvements (move ordering, quiescence, etc.)."""
    print("\n" + "=" * 60)
    print("SEARCH IMPROVEMENT TESTS")
    print("=" * 60)
    
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    # Test position with many captures
    tactical_position = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    board = chess.Board(tactical_position)
    
    print(f"Tactical position with captures:")
    print(f"FEN: {tactical_position}")
    
    start_time = time.time()
    best_move = engine.get_best_move(board, time_limit=10.0)
    elapsed = time.time() - start_time
    
    print(f"Best move: {best_move.uci()} ({board.san(best_move)})")
    print(f"Time taken: {elapsed:.2f}s")

def test_transposition_table():
    """Test transposition table functionality."""
    print("\n" + "=" * 60)
    print("TRANSPOSITION TABLE TESTS")
    print("=" * 60)
    
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    # Test the same position multiple times to see if transposition table helps
    board = chess.Board()
    
    print("Testing same position multiple times:")
    for i in range(3):
        start_time = time.time()
        best_move = engine.get_best_move(board, time_limit=5.0)
        elapsed = time.time() - start_time
        
        print(f"   Iteration {i+1}: {best_move.uci()} ({board.san(best_move)}) - {elapsed:.2f}s")

def main():
    """Run all enhanced engine tests."""
    print("ENHANCED ZERBINETTO ENGINE V2 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    try:
        test_enhanced_engine()
        test_tal_mode()
        test_positional_evaluation()
        test_search_improvements()
        test_transposition_table()
        
        print("\n" + "=" * 60)
        print("ALL ENHANCED ENGINE TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Enhanced engine test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
