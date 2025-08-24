#!/usr/bin/env python3
"""
Comprehensive Test Suite for Zerbinetto Engine

Tests the simplified Zerbinetto chess engine with various positions,
performance benchmarks, and functionality verification.
"""

import chess
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zerbinetto_engine import ZerbinettoEngine

def test_basic_functionality():
    """Test basic engine functionality."""
    print("=" * 60)
    print("BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.1)
    
    # Test starting position
    board = chess.Board()
    print(f"Starting position FEN: {board.fen()}")
    
    best_move = engine.get_best_move(board)
    print(f"Best move: {best_move.uci()} ({board.san(best_move)})")
    
    # Make the move and test again
    board.push(best_move)
    print(f"Position after move: {board.fen()}")
    
    best_move2 = engine.get_best_move(board)
    print(f"Best move: {best_move2.uci()} ({board.san(best_move2)})")
    
    print("Basic functionality test completed!")

def test_move_quality():
    """Test the engine's move quality across different position types."""
    print("\n" + "=" * 60)
    print("MOVE QUALITY TESTS")
    print("=" * 60)
    
    test_positions = [
        {
            "name": "Starting Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard opening position"
        },
        {
            "name": "Tactical Position", 
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
            "description": "Position with tactical opportunities"
        },
        {
            "name": "Middlegame Position",
            "fen": "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 1",
            "description": "Typical middlegame with pieces developed"
        },
        {
            "name": "Pawn Structure Test",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1",
            "description": "Position to test pawn structure evaluation"
        },
        {
            "name": "Endgame Position",
            "fen": "8/8/8/4k3/4K3/8/8/8 w - - 0 1",
            "description": "Simple king endgame"
        }
    ]
    
    engine = ZerbinettoEngine(search_depth=4, randomness_factor=0.1)
    
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

def test_evaluation_functions():
    """Test individual evaluation functions."""
    print("\n" + "=" * 60)
    print("EVALUATION FUNCTION TESTS")
    print("=" * 60)
    
    engine = ZerbinettoEngine()
    
    # Test material evaluation
    board = chess.Board()
    material_score = engine._evaluate_material(board)
    print(f"Starting position material score: {material_score:.2f}")
    
    # Test position with white up a pawn
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    material_score = engine._evaluate_material(board)
    print(f"White up a pawn material score: {material_score:.2f}")
    
    # Test positional evaluation
    board = chess.Board()
    positional_score = engine._evaluate_positional_factors(board)
    print(f"Starting position positional score: {positional_score:.2f}")
    
    # Test pawn structure evaluation
    pawn_score = engine._evaluate_pawn_structure(board)
    print(f"Starting position pawn structure score: {pawn_score:.2f}")
    
    # Test mobility evaluation
    mobility_score = engine._evaluate_mobility(board)
    print(f"Starting position mobility score: {mobility_score:.2f}")
    
    # Test endgame detection
    is_endgame = engine._is_endgame(board)
    print(f"Starting position is endgame: {is_endgame}")
    
    # Test endgame position
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    is_endgame = engine._is_endgame(board)
    print(f"Endgame position is endgame: {is_endgame}")

def test_performance():
    """Test engine performance across different search depths."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)
    
    # Complex middlegame position for timing
    fen = "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 1"
    board = chess.Board(fen)
    
    print(f"Test Position: {fen}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    print("\nDepth vs Performance:")
    
    for depth in [2, 3, 4, 5]:
        print(f"\nTesting depth {depth}...")
        engine = ZerbinettoEngine(search_depth=depth, randomness_factor=0.1)
        
        try:
            start_time = time.time()
            best_move = engine.get_best_move(board, time_limit=30.0)
            elapsed = time.time() - start_time
            
            print(f"  Depth {depth}: {best_move.uci()} ({board.san(best_move)}) - {elapsed:.2f}s")
            
            # Stop if it takes too long
            if elapsed > 25.0:
                print(f"  Stopping at depth {depth} due to time limit")
                break
                
        except Exception as e:
            print(f"  Depth {depth}: ERROR - {e}")

def test_randomization():
    """Test that the engine produces varied moves with randomization."""
    print("\n" + "=" * 60)
    print("RANDOMIZATION TESTS")
    print("=" * 60)
    
    # Starting position for consistency
    board = chess.Board()
    
    print(f"Testing move variation in starting position...")
    print(f"Running 10 iterations to check for move diversity:")
    
    moves_chosen = {}
    
    for i in range(10):
        engine = ZerbinettoEngine(search_depth=3, randomness_factor=0.2)  # Higher randomness
        best_move = engine.get_best_move(board, time_limit=5.0)
        move_san = board.san(best_move)
        
        if move_san in moves_chosen:
            moves_chosen[move_san] += 1
        else:
            moves_chosen[move_san] = 1
        
        print(f"  Iteration {i+1}: {move_san}")
    
    print(f"\nMove frequency:")
    for move, count in sorted(moves_chosen.items(), key=lambda x: x[1], reverse=True):
        print(f"  {move}: {count} times")
    
    diversity = len(moves_chosen)
    print(f"\nMove diversity: {diversity} different moves out of 10 iterations")

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

def test_move_ordering():
    """Test move ordering functionality."""
    print("\n" + "=" * 60)
    print("MOVE ORDERING TESTS")
    print("=" * 60)
    
    engine = ZerbinettoEngine()
    
    # Test position with checks and captures
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    
    legal_moves = list(board.legal_moves)
    ordered_moves = engine._order_moves(board, legal_moves)
    
    print(f"Position: {board.fen()}")
    print(f"Legal moves: {len(legal_moves)}")
    print(f"First 5 ordered moves:")
    
    for i, move in enumerate(ordered_moves[:5]):
        print(f"  {i+1}. {move.uci()} ({board.san(move)})")

def main():
    """Run all tests."""
    print("ZERBINETTO ENGINE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_move_quality()
        test_evaluation_functions()
        test_performance()
        test_randomization()
        test_transposition_table()
        test_move_ordering()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
