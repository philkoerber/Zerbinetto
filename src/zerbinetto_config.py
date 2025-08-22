"""
Zerbinetto Engine Configuration

Configuration parameters for the Zerbinetto chess engine.
"""

# Search parameters
SEARCH_DEPTH = 4  # Depth for minimax search (balanced for strength vs speed)
RANDOMNESS_FACTOR = 0.15  # Factor for move randomization (0.0-1.0)

# Tactical bias weights (in pawns)
BIAS_WEIGHTS = {
    'sacrifice': 0.05,          # Small bonus for sacrifices (reduced from 0.3)
    'open_lines': 0.08,         # Bonus for opening lines (reduced from 0.15)
    'king_attack': 0.12,        # Bonus for king attacks (reduced from 0.25)
    'imbalance': 0.1,           # Bonus for imbalanced positions (reduced from 0.2)
    'simplification': -0.05,    # Penalty for simplifying (reduced from -0.1)
    'own_king_exposure': -0.1,  # Increased penalty for own king exposure
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
        [-0.1,  0.05, 0.05, 0.05, 0.05, 0.05, 0.0,-0.1],
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

# King safety evaluation
KING_SAFETY_WEIGHTS = {
    'pawn_shield': 0.1,         # Bonus for pawn shield
    'king_distance_center': 0.05, # Penalty for king being central
    'enemy_attackers': -0.1,    # Penalty per enemy piece attacking king area
    'castled': 0.2,             # Bonus for castled king
}

# Tal mode configuration
TAL_MODE_CONFIG = {
    'enabled': True,            # Enable Tal mode
    'frequency': 0.15,          # 15% chance per game to enter Tal mode
    'sacrifice_threshold': 0.5, # Maximum material to sacrifice in Tal mode (in pawns)
    'flashy_bonus': 0.3,        # Bonus for flashy moves in Tal mode
}

# Engine behavior settings
ENGINE_SETTINGS = {
    'max_move_time': 8.0,       # Maximum time to think about a move (seconds) - reasonable for online play
    'move_delay': 1.0,          # Delay before making a move (seconds)
    'enable_debug_logging': True,  # Enable detailed move evaluation logging
    'quiescence_depth': 5,      # Depth for quiescence search
    'transposition_table_size': 10000,  # Size of transposition table
}

# Style constraints
STYLE_CONSTRAINTS = {
    'avoid_trivial_blunders': True,  # Basic blunder check before committing to move
    'prioritize_chaos': True,        # Prefer chaotic middlegames over calm positions
    'prefer_counterplay': True,      # In lost positions, prefer counterplay over passive defense
    'avoid_grinding': True,          # In won positions, avoid grinding endgames
}
