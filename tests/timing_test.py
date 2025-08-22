#!/usr/bin/env python3
"""
Timing test for the Zerbinetto engine

Tests how long the engine takes to think with different search depths.
"""

import chess
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zerbinetto_engine import ZerbinettoEngine

def test_timing():
    """Test engine timing with different depths."""
    print("Testing Zerbinetto Engine Timing")
    print("=" * 50)
    
    # Test position: complex middlegame
    fen = "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 1"
    board = chess.Board(fen)
    
    print(f"Position: {fen}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    print()
    
    # Test different depths
    for depth in [2, 3, 4, 5, 6]:
        print(f"Testing depth {depth}...")
        engine = ZerbinettoEngine(search_depth=depth, randomness_factor=0.1)
        
        start_time = time.time()
        best_move = engine.get_best_move(board, time_limit=30.0)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"  Depth {depth}: {best_move.uci()} ({board.san(best_move)}) - {elapsed:.2f}s")
        
        # Stop if it takes too long
        if elapsed > 20.0:
            print(f"  Stopping at depth {depth} due to time limit")
            break
    
    print("\n" + "=" * 50)
    print("Timing test completed!")

if __name__ == "__main__":
    test_timing()
