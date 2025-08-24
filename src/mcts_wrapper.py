#!/usr/bin/env python3
"""
MCTS Wrapper for ML Engine

A simple Monte Carlo Tree Search wrapper that can be used with the ML engine
for better move selection.
"""

import chess
import logging
import math
import random
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .ml_engine import MLEngine

logger = logging.getLogger(__name__)

class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, board: chess.Board, parent=None, move=None):
        """Initialize an MCTS node.
        
        Args:
            board: Chess position at this node
            parent: Parent node
            move: Move that led to this position
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}  # move -> MCTSNode
        self.visits = 0
        self.value = 0.0
        self.untried_moves = list(board.legal_moves)
    
    def is_fully_expanded(self) -> bool:
        """Check if all children have been expanded."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.board.is_game_over()
    
    def get_ucb_value(self, exploration_constant: float = 1.414) -> float:
        """Get UCB value for this node.
        
        Args:
            exploration_constant: UCB exploration constant
            
        Returns:
            UCB value
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select the best child using UCB.
        
        Args:
            exploration_constant: UCB exploration constant
            
        Returns:
            Best child node
        """
        return max(self.children.values(), key=lambda child: child.get_ucb_value(exploration_constant))
    
    def expand(self) -> 'MCTSNode':
        """Expand an untried move.
        
        Returns:
            New child node
        """
        if not self.untried_moves:
            return self
        
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        new_board = self.board.copy()
        new_board.push(move)
        
        child = MCTSNode(new_board, parent=self, move=move)
        self.children[move] = child
        
        return child
    
    def simulate(self, ml_engine: MLEngine, max_moves: int = 50) -> float:
        """Simulate a random playout from this position.
        
        Args:
            ml_engine: ML engine for move selection
            max_moves: Maximum moves to simulate
            
        Returns:
            Game result (1.0 for white win, 0.0 for black win, 0.5 for draw)
        """
        board = self.board.copy()
        moves_played = 0
        
        while not board.is_game_over() and moves_played < max_moves:
            try:
                # Use ML engine's direct prediction (not MCTS) to avoid recursion
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                
                # Get move scores using direct model prediction
                move_scores = []
                for move in legal_moves:
                    board.push(move)
                    features = ml_engine.position_encoder.encode_position(board)
                    score = ml_engine.model.predict(features)
                    board.pop()
                    move_scores.append((move, score))
                
                # Sort by score and select best move
                if board.turn == chess.WHITE:
                    move_scores.sort(key=lambda x: x[1], reverse=True)
                else:
                    move_scores.sort(key=lambda x: x[1], reverse=False)
                
                best_move = move_scores[0][0]
                board.push(best_move)
                moves_played += 1
                
            except Exception as e:
                logger.debug(f"Error in simulation: {e}")
                # Fallback to random move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    board.push(move)
                    moves_played += 1
                else:
                    break
        
        # Determine result
        if board.is_checkmate():
            # Last player to move won
            return 1.0 if board.turn == chess.BLACK else 0.0
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.5  # Draw
        else:
            return 0.5  # Draw (max moves reached)
    
    def backpropagate(self, result: float):
        """Backpropagate simulation result up the tree.
        
        Args:
            result: Game result from white's perspective
        """
        node = self
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent
    
    def get_best_move(self) -> chess.Move:
        """Get the best move based on visit counts.
        
        Returns:
            Best move
        """
        if not self.children:
            return None
        
        return max(self.children.keys(), key=lambda move: self.children[move].visits)

class MCTSWrapper:
    """Monte Carlo Tree Search wrapper for the ML engine."""
    
    def __init__(self, ml_engine: MLEngine, iterations: int = 1000, time_limit: float = 5.0):
        """Initialize the MCTS wrapper.
        
        Args:
            ml_engine: ML engine for move evaluation
            iterations: Number of MCTS iterations
            time_limit: Time limit for MCTS search (seconds)
        """
        self.ml_engine = ml_engine
        self.iterations = iterations
        self.time_limit = time_limit
    
    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose the best move using MCTS.
        
        Args:
            board: Current chess position
            
        Returns:
            Best move found
        """
        if board.is_game_over():
            raise ValueError("No legal moves available")
        
        # Create root node
        root = MCTSNode(board)
        
        # Run MCTS iterations
        start_time = time.time()
        iteration = 0
        
        while iteration < self.iterations and (time.time() - start_time) < self.time_limit:
            # Selection
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child()
            
            # Expansion
            if not node.is_terminal():
                node = node.expand()
            
            # Simulation
            result = node.simulate(self.ml_engine)
            
            # Backpropagation
            node.backpropagate(result)
            
            iteration += 1
        
        # Get best move
        best_move = root.get_best_move()
        
        if best_move is None:
            # Fallback to ML engine directly
            logger.warning("MCTS failed to find move, falling back to ML engine")
            return self.ml_engine.choose_move(board)
        
        logger.info(f"MCTS selected move: {best_move.uci()} after {iteration} iterations")
        return best_move
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a position using MCTS.
        
        Args:
            board: Position to evaluate
            
        Returns:
            Position evaluation score
        """
        # For evaluation, we can use the ML engine directly
        return self.ml_engine.evaluate_position(board)
