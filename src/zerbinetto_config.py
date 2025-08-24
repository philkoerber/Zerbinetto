"""
Zerbinetto Engine Configuration

Simple, solid configuration for the Zerbinetto chess engine.
"""

# Search parameters
SEARCH_DEPTH = 4  # Depth for minimax search
RANDOMNESS_FACTOR = 0.1  # Factor for move randomization (0.0-1.0)

# Piece values for material evaluation
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
    'mobility': 0.02,           # Bonus per legal move
    'pawn_structure': 0.15,     # Weight for pawn structure evaluation
    'piece_square_tables': 0.1, # Weight for piece-square table bonuses
}

# Piece-square tables (positional bonuses for each piece type)
PIECE_SQUARE_TABLES = {
    'pawn': [
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
        [0.1,  0.1,  0.2,  0.3,  0.3,  0.2,  0.1,  0.1],
        [0.05, 0.05, 0.1,  0.25, 0.25, 0.1,  0.05, 0.05],
        [0.0,  0.0,  0.0,  0.2,  0.2,  0.0,  0.0,  0.0],
        [0.05,-0.05,-0.1,  0.0,  0.0,-0.1,-0.05, 0.05],
        [0.05, 0.1,  0.1,-0.2,-0.2, 0.1,  0.1,  0.05],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
    ],
    'knight': [
        [-0.5,-0.4,-0.3,-0.3,-0.3,-0.3,-0.4,-0.5],
        [-0.4,-0.2,  0.0,  0.0,  0.0,  0.0,-0.2,-0.4],
        [-0.3,  0.0,  0.1,  0.15, 0.15, 0.1,  0.0,-0.3],
        [-0.3,  0.05, 0.15, 0.2,  0.2,  0.15, 0.05,-0.3],
        [-0.3,  0.0,  0.15, 0.2,  0.2,  0.15, 0.0,-0.3],
        [-0.3,  0.05, 0.1,  0.15, 0.15, 0.1,  0.05,-0.3],
        [-0.4,-0.2,  0.0,  0.05, 0.05, 0.0,-0.2,-0.4],
        [-0.5,-0.4,-0.3,-0.3,-0.3,-0.3,-0.4,-0.5]
    ],
    'bishop': [
        [-0.2,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.2],
        [-0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,-0.1],
        [-0.1,  0.0,  0.05, 0.1,  0.1,  0.05, 0.0,-0.1],
        [-0.1,  0.05, 0.05, 0.1,  0.1,  0.05, 0.05,-0.1],
        [-0.1,  0.0,  0.1,  0.1,  0.1,  0.1,  0.0,-0.1],
        [-0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,-0.1],
        [-0.1,  0.05, 0.0,  0.0,  0.0,  0.0, 0.05,-0.1],
        [-0.2,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.2]
    ],
    'rook': [
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.05, 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.05],
        [-0.05, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,-0.05],
        [-0.05, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,-0.05],
        [-0.05, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,-0.05],
        [-0.05, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,-0.05],
        [-0.05, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,-0.05],
        [0.0,  0.0,  0.0,  0.05, 0.05, 0.0,  0.0,  0.0]
    ],
    'queen': [
        [-0.2,-0.1,-0.1,-0.05,-0.05,-0.1,-0.1,-0.2],
        [-0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,-0.1],
        [-0.1,  0.0,  0.05, 0.05, 0.05, 0.05, 0.0,-0.1],
        [-0.05, 0.0,  0.05, 0.05, 0.05, 0.05, 0.0,-0.05],
        [0.0,  0.0,  0.05, 0.05, 0.05, 0.05, 0.0,-0.05],
        [-0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0,-0.1],
        [-0.1,  0.0,  0.05, 0.0,  0.0,  0.0,  0.0,-0.1],
        [-0.2,-0.1,-0.1,-0.05,-0.05,-0.1,-0.1,-0.2]
    ],
    'king_middlegame': [
        [-0.3,-0.4,-0.4,-0.5,-0.5,-0.4,-0.4,-0.3],
        [-0.3,-0.4,-0.4,-0.5,-0.5,-0.4,-0.4,-0.3],
        [-0.3,-0.4,-0.4,-0.5,-0.5,-0.4,-0.4,-0.3],
        [-0.3,-0.4,-0.4,-0.5,-0.5,-0.4,-0.4,-0.3],
        [-0.2,-0.3,-0.3,-0.4,-0.4,-0.3,-0.3,-0.2],
        [-0.1,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.1],
        [0.2,  0.2,  0.0,  0.0,  0.0,  0.0,  0.2,  0.2],
        [0.2,  0.3,  0.1,  0.0,  0.0,  0.1,  0.3,  0.2]
    ],
    'king_endgame': [
        [-0.5,-0.4,-0.3,-0.2,-0.2,-0.3,-0.4,-0.5],
        [-0.3,-0.2,-0.1,  0.0,  0.0,-0.1,-0.2,-0.3],
        [-0.3,-0.1,  0.2,  0.3,  0.3,  0.2,-0.1,-0.3],
        [-0.3,-0.1,  0.3,  0.4,  0.4,  0.3,-0.1,-0.3],
        [-0.3,-0.1,  0.3,  0.4,  0.4,  0.3,-0.1,-0.3],
        [-0.3,-0.1,  0.2,  0.3,  0.3,  0.2,-0.1,-0.3],
        [-0.3,-0.3,  0.0,  0.0,  0.0,  0.0,-0.3,-0.3],
        [-0.5,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.5]
    ]
}

# Pawn structure evaluation weights
PAWN_STRUCTURE_WEIGHTS = {
    'isolated_pawn': -0.2,      # Penalty for isolated pawns
    'doubled_pawn': -0.15,      # Penalty for doubled pawns
    'passed_pawn': 0.3,         # Bonus for passed pawns
    'backward_pawn': -0.1,      # Penalty for backward pawns
    'connected_pawns': 0.05,    # Bonus for connected pawns
}

# Engine behavior settings
ENGINE_SETTINGS = {
    'max_move_time': 8.0,       # Maximum time to think about a move (seconds)
    'move_delay': 1.0,          # Delay before making a move (seconds)
    'enable_debug_logging': True,  # Enable detailed move evaluation logging
    'quiescence_depth': 3,      # Depth for quiescence search
    'transposition_table_size': 10000,  # Size of transposition table
}
