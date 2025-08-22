"""
Tal Engine Configuration

Configuration parameters for the Tal-style chess engine.
"""

# Search parameters
SEARCH_DEPTH = 3  # Depth for minimax search (3-4 plies recommended)
RANDOMNESS_FACTOR = 0.15  # Factor for move randomization (0.0-1.0)

# Tactical bias weights (in pawns)
BIAS_WEIGHTS = {
    'sacrifice': 0.3,           # Bonus for sacrifices
    'open_lines': 0.15,         # Bonus for opening lines
    'king_attack': 0.25,        # Bonus for king attacks
    'imbalance': 0.2,           # Bonus for imbalanced positions
    'simplification': -0.1,     # Penalty for simplifying
    'own_king_exposure': -0.05, # Small penalty for own king exposure
}

# Piece values for basic evaluation
PIECE_VALUES = {
    'pawn': 1.0,
    'knight': 3.0,
    'bishop': 3.2,
    'rook': 5.0,
    'queen': 9.0,
    'king': 1000.0  # Very high to avoid king captures
}

# Position evaluation weights
POSITION_WEIGHTS = {
    'center_control': 0.1,      # Bonus for controlling center squares
    'king_safety': 0.05,        # Penalty for king being too central
}

# Engine behavior settings
ENGINE_SETTINGS = {
    'max_move_time': 10.0,      # Maximum time to think about a move (seconds)
    'move_delay': 1.0,          # Delay before making a move (seconds)
    'enable_debug_logging': True,  # Enable detailed move evaluation logging
}

# Style constraints
STYLE_CONSTRAINTS = {
    'avoid_trivial_blunders': True,  # Basic blunder check before committing to move
    'prioritize_chaos': True,        # Prefer chaotic middlegames over calm positions
    'prefer_counterplay': True,      # In lost positions, prefer counterplay over passive defense
    'avoid_grinding': True,          # In won positions, avoid grinding endgames
}
