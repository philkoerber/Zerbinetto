"""
Game Handler

Handles chess game logic, legal move generation, and game state management.
"""

import asyncio
import json
import logging
import random
import time
from typing import Dict, List, Optional, Tuple
import aiohttp
import chess
import chess.engine

from ml_engine import MLEngine

logger = logging.getLogger(__name__)

class GameHandler:
    """Handles chess game logic and move generation."""
    
    def __init__(self, lichess_client):
        """Initialize the game handler.
        
        Args:
            lichess_client: LichessClient instance for making moves
        """
        self.lichess_client = lichess_client
        self.active_games: Dict[str, Dict] = {}
        self.game_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize ML engine
        self.ml_engine = MLEngine()
        
        # Configuration
        self.move_delay = 1.0  # Delay before making a move (seconds)
        self.max_move_time = 8.0  # Maximum time to think about a move (seconds)
    
    async def handle_game_start(self, game_data: Dict):
        """Handle when a game starts.
        
        Args:
            game_data: Game start event data
        """
        game_id = game_data.get('game', {}).get('id')
        if not game_id:
            logger.warning("Game start event missing game ID")
            return
        
        logger.info(f"Game started: {game_id}")
        
        # Store game info including our color
        our_color = game_data.get('game', {}).get('color', 'unknown')
        self.active_games[game_id] = {
            'id': game_id,
            'started_at': time.time(),
            'moves': [],
            'status': 'playing',
            'our_color': our_color  # 'white' or 'black'
        }
        
        # Store game ID for future reference
        # self.ml_engine.game_id = game_id  # ML engine doesn't need game_id
        
        logger.info(f"We are playing as {our_color} in game {game_id}")
        
        # Check if it's our turn to move first
        is_our_turn = game_data.get('game', {}).get('isMyTurn', False)
        logger.info(f"Game {game_id} - isMyTurn: {is_our_turn}")
        
        if is_our_turn:
            logger.info(f"It's our turn to move first in game {game_id}")
            # Make a move immediately
            await asyncio.sleep(1.0)  # Small delay
            move = await self._generate_ml_move(game_data)
            if move:
                logger.info(f"Making first move {move} in game {game_id}")
                await self.lichess_client.make_move(game_id, move)
                logger.info(f"Successfully made first move {move} in game {game_id}")
        else:
            logger.info(f"Not our turn to move first in game {game_id}")
        
        # Start monitoring this game for state changes
        # Try streaming first, fallback to polling if it fails
        self.game_tasks[game_id] = asyncio.create_task(self._monitor_game_with_fallback(game_id))
    
    async def _try_make_move_with_retries(self, game_id: str, game_data: Dict) -> bool:
        """Try to make a move with retries if the first attempt fails.
        
        Args:
            game_id: ID of the game
            game_data: Current game state
            
        Returns:
            True if a move was successfully made, False otherwise
        """
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                # Generate a legal move
                move = await self._generate_ml_move(game_data)
                if not move:
                    logger.warning(f"No legal move generated on attempt {attempt + 1}")
                    continue
                
                logger.info(f"Attempt {attempt + 1}: Making move {move} in game {game_id}")
                
                # Try to make the move
                await self.lichess_client.make_move(game_id, move)
                logger.info(f"Successfully made move {move} in game {game_id}")
                return True
                
            except Exception as e:
                logger.warning(f"Move attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # Short delay before retry
                    continue
                else:
                    logger.error(f"All {max_retries} move attempts failed for game {game_id}")
                    return False
        
        return False
    
    async def _monitor_game(self, game_id: str):
        """Monitor a game for state changes and make moves when it's our turn.
        
        Args:
            game_id: ID of the game to monitor
        """
        try:
            logger.info(f"Starting to monitor game {game_id}")
            
            # Use the bot game stream endpoint
            url = f"https://lichess.org/api/bot/game/stream/{game_id}"
            headers = {
                "Authorization": f"Bearer {self.lichess_client.token}",
                "User-Agent": "ZerbinettoBot/1.0 (https://github.com/yourusername/zerbinetto)"
            }
            
            logger.info(f"Connecting to game stream: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    logger.info(f"Game stream response status: {response.status}")
                    response.raise_for_status()
                    logger.info(f"Connected to game stream for {game_id}")
                    
                    async for line in response.content:
                        if line:
                            try:
                                line_str = line.decode('utf-8').strip()
                                if line_str:
                                    data = json.loads(line_str)
                                    await self._handle_game_stream_event(game_id, data)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON from game {game_id}: {line}")
                            except Exception as e:
                                logger.error(f"Error handling game stream event for {game_id}: {e}", exc_info=True)
                                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error monitoring game {game_id}: {e}")
        except Exception as e:
            logger.error(f"Error monitoring game {game_id}: {e}", exc_info=True)
        finally:
            if game_id in self.game_tasks:
                del self.game_tasks[game_id]
            logger.info(f"Stopped monitoring game {game_id}")
    
    async def _monitor_game_with_fallback(self, game_id: str):
        """Monitor a game with fallback to polling if streaming fails.
        
        Args:
            game_id: ID of the game to monitor
        """
        try:
            # Try streaming first
            await self._monitor_game(game_id)
        except Exception as e:
            logger.warning(f"Game stream failed for {game_id}, falling back to polling: {e}")
            # Fallback to polling
            await self._poll_game_state(game_id)
    
    async def _poll_game_state(self, game_id: str):
        """Poll game state as a fallback when streaming fails.
        
        Args:
            game_id: ID of the game to poll
        """
        try:
            logger.info(f"Starting to poll game state for {game_id}")
            
            while game_id in self.active_games and self.active_games[game_id]['status'] == 'playing':
                try:
                    # Get current game state
                    game_state = await self.lichess_client.get_game_state(game_id)
                    if game_state:
                        # Check if it's our turn
                        moves = game_state.get('moves', '').split()
                        if self._is_our_turn_by_moves(game_id, moves):
                            logger.info(f"Polling detected it's our turn for game {game_id}")
                            await self._handle_game_state_update(game_id, game_state)
                    
                    # Wait before polling again
                    await asyncio.sleep(3.0)  # Poll every 3 seconds
                    
                except Exception as e:
                    logger.error(f"Error polling game {game_id}: {e}")
                    await asyncio.sleep(5.0)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Error in polling for game {game_id}: {e}")
        finally:
            logger.info(f"Stopped polling game {game_id}")
    
    async def _handle_game_stream_event(self, game_id: str, data: Dict):
        """Handle events from a game stream.
        
        Args:
            game_id: ID of the game
            data: Event data
        """
        event_type = data.get('type')
        logger.info(f"Game {game_id} stream event: {event_type} - {json.dumps(data, indent=2)}")
        
        if event_type == 'gameFull':
            # Initial game state
            await self._handle_game_state_update(game_id, data)
        elif event_type == 'gameState':
            # Game state update (move made)
            await self._handle_game_state_update(game_id, data)
        elif event_type == 'chatLine':
            # Chat message (ignore for now)
            pass
        else:
            logger.debug(f"Unhandled game stream event type: {event_type}")
    
    async def _handle_game_state_update(self, game_id: str, data: Dict):
        """Handle game state updates and make moves if it's our turn.
        
        Args:
            game_id: ID of the game
            data: Game state data
        """
        # Get move history
        moves = data.get('moves', '').split()
        status = data.get('status', 'started')
        
        logger.info(f"Game {game_id} state update: {len(moves)} moves, status: {status}")
        
        if status != 'started':
            logger.info(f"Game {game_id} is not in started state: {status}")
            return
        
        # Update our game record
        if game_id in self.active_games:
            self.active_games[game_id]['moves'] = moves
            self.active_games[game_id]['status'] = status
        
        # Check if it's our turn using the move count
        if self._is_our_turn_by_moves(game_id, moves):
            logger.info(f"It's our turn in game {game_id}! Making move...")
            try:
                await asyncio.sleep(self.move_delay)
                
                # Try to make a move with retries
                success = await self._try_make_move_with_retries(game_id, data)
                if success:
                    logger.info(f"Successfully made a move in game {game_id}")
                else:
                    logger.warning(f"Failed to make any move in game {game_id}")
            except Exception as e:
                logger.error(f"Error making move in game {game_id}: {e}", exc_info=True)
        else:
            logger.info(f"Not our turn in game {game_id}")
    
    def _is_our_turn_by_moves(self, game_id: str, moves: List[str]) -> bool:
        """Check if it's our turn based on move count and our color.
        
        Args:
            game_id: ID of the game
            moves: List of moves made so far
            
        Returns:
            True if it's our turn
        """
        if game_id not in self.active_games:
            return False
        
        game = self.active_games[game_id]
        move_count = len(moves)
        our_color = game.get('our_color', 'unknown')
        
        logger.debug(f"Game {game_id}: {move_count} moves, we are {our_color}")
        
        # Even move count (0, 2, 4...) = white's turn
        # Odd move count (1, 3, 5...) = black's turn
        white_turn = move_count % 2 == 0
        
        if our_color == 'white':
            is_our_turn = white_turn
        elif our_color == 'black':
            is_our_turn = not white_turn
        else:
            logger.warning(f"Unknown color for game {game_id}: {our_color}")
            return False
        
        logger.debug(f"Game {game_id}: white_turn={white_turn}, our_color={our_color}, is_our_turn={is_our_turn}")
        return is_our_turn
    
    async def handle_game_finish(self, game_data: Dict):
        """Handle when a game finishes.
        
        Args:
            game_data: Game finish event data
        """
        game_id = game_data.get('game', {}).get('id')
        if not game_id:
            logger.warning("Game finish event missing game ID")
            return
        
        logger.info(f"Game finished: {game_id}")
        
        # Clean up game
        if game_id in self.active_games:
            self.active_games[game_id]['status'] = 'finished'
            self.active_games[game_id]['result'] = game_data.get('game', {}).get('status')
        
        # Cancel the game monitoring task
        if game_id in self.game_tasks:
            self.game_tasks[game_id].cancel()
            del self.game_tasks[game_id]
            logger.info(f"Cancelled monitoring task for game {game_id}")
    
    async def handle_game_state(self, game_data: Dict):
        """Handle game state updates.
        
        Args:
            game_data: Game state event data
        """
        game_id = game_data.get('id')
        if not game_id:
            logger.warning("Game state event missing game ID")
            return
        
        # Debug: Log the full game data to understand what we're receiving
        logger.info(f"Received game state for {game_id}: {json.dumps(game_data, indent=2)}")
        
        if game_id not in self.active_games:
            logger.warning(f"Received game state for unknown game: {game_id}")
            # Try to create the game entry if it doesn't exist
            self.active_games[game_id] = {
                'id': game_id,
                'started_at': time.time(),
                'moves': [],
                'status': 'playing'
            }
        
        # Update game state
        game = self.active_games[game_id]
        game['moves'] = game_data.get('moves', '').split()
        game['status'] = game_data.get('status')
        
        # Store additional game info
        if 'white' in game_data:
            game['white'] = game_data['white']
        if 'black' in game_data:
            game['black'] = game_data['black']
        
        # For now, we'll only handle the first move
        # TODO: Add proper game state polling for subsequent moves
        logger.info(f"Game state update received for {game_id}")
    

    
    def _is_our_turn(self, game_data: Dict) -> bool:
        """Check if it's our turn to move.
        
        Args:
            game_data: Game state data
            
        Returns:
            True if it's our turn, False otherwise
        """
        status = game_data.get('status')
        if status != 'started':
            logger.debug(f"Game not started, status: {status}")
            return False
        
        # Get move history
        moves = game_data.get('moves', '').split()
        move_count = len(moves)
        
        # Get our bot's username
        bot_username = 'zerbinetto'  # Our bot's username (lowercase)
        
        # Get player colors
        white_player = game_data.get('white', {}).get('id', '').lower()
        black_player = game_data.get('black', {}).get('id', '').lower()
        
        logger.debug(f"Move count: {move_count}, White: {white_player}, Black: {black_player}, Bot: {bot_username}")
        
        # Determine if we're white or black
        if bot_username == white_player:
            # We're white - it's our turn if move count is even (0, 2, 4, ...)
            our_turn = move_count % 2 == 0
            logger.debug(f"We are white, our turn: {our_turn}")
            return our_turn
        elif bot_username == black_player:
            # We're black - it's our turn if move count is odd (1, 3, 5, ...)
            our_turn = move_count % 2 == 1
            logger.debug(f"We are black, our turn: {our_turn}")
            return our_turn
        else:
            logger.warning(f"Bot username '{bot_username}' not found in players: white={white_player}, black={black_player}")
            return False
    
    async def _make_move(self, game_id: str, game_data: Dict):
        """Make a move in the game.
        
        Args:
            game_id: ID of the game
            game_data: Current game state
        """
        try:
            logger.info(f"Starting to make move for game {game_id}")
            
            # Add a small delay to avoid making moves too quickly
            await asyncio.sleep(self.move_delay)
            
            # Generate an ML-based move
            logger.info(f"Generating ML-based move for game {game_id}")
            move = await self._generate_ml_move(game_data)
            
            if move:
                logger.info(f"Generated move {move} for game {game_id}, sending to Lichess...")
                await self.lichess_client.make_move(game_id, move)
                logger.info(f"Successfully made move {move} in game {game_id}")
            else:
                logger.warning(f"No legal moves available in game {game_id}")
                
        except Exception as e:
            logger.error(f"Error making move in game {game_id}: {e}", exc_info=True)
    
    async def _generate_ml_move(self, game_data: Dict) -> Optional[str]:
        """Generate a move using the ML engine.
        
        Args:
            game_data: Current game state
            
        Returns:
            A legal move in UCI format, or None if no moves available
        """
        logger.info("Generating ML-based move...")
        
        # Get the current FEN position - handle both direct and nested game data
        fen = game_data.get('fen', '')
        if not fen:
            fen = game_data.get('game', {}).get('fen', '')
        
        logger.info(f"Current FEN: {fen}")
        
        # If no FEN provided, try to build it from moves
        if not fen:
            moves = game_data.get('moves', '')
            if not moves:
                moves = game_data.get('state', {}).get('moves', '')
            
            if moves:
                logger.info(f"Building FEN from moves: {moves}")
                try:
                    # Create a board and play the moves
                    board = chess.Board()
                    move_list = moves.split()
                    for move_uci in move_list:
                        move = chess.Move.from_uci(move_uci)
                        board.push(move)
                    
                    fen = board.fen()
                    logger.info(f"Built FEN: {fen}")
                except Exception as e:
                    logger.error(f"Error building FEN from moves: {e}")
                    return None
            else:
                logger.warning("No FEN position or moves provided")
                return None
        
        try:
            # Create a chess board from the FEN position
            board = chess.Board(fen)
            
            # Get the best move from ML engine
            best_move = self.ml_engine.choose_move(board)
            
            if best_move:
                move_uci = best_move.uci()
                logger.info(f"ML engine selected move: {move_uci}")
                return move_uci
            else:
                logger.warning("ML engine returned no move")
                return None
            
        except Exception as e:
            logger.error(f"Error generating ML move: {e}", exc_info=True)
            # Fallback to random move if ML engine fails
            logger.info("Falling back to random move due to ML engine error")
            return await self._generate_random_move_fallback(game_data)
    
    async def _generate_random_move_fallback(self, game_data: Dict) -> Optional[str]:
        """Fallback method to generate a random legal move.
        
        Args:
            game_data: Current game state
            
        Returns:
            A legal move in UCI format, or None if no moves available
        """
        logger.info("Generating random legal move (fallback)...")
        
        # Get the current FEN position
        fen = game_data.get('fen', '')
        if not fen:
            fen = game_data.get('game', {}).get('fen', '')
        
        # If no FEN provided, try to build it from moves
        if not fen:
            moves = game_data.get('moves', '')
            if not moves:
                moves = game_data.get('state', {}).get('moves', '')
            
            if moves:
                try:
                    board = chess.Board()
                    move_list = moves.split()
                    for move_uci in move_list:
                        move = chess.Move.from_uci(move_uci)
                        board.push(move)
                    fen = board.fen()
                except Exception as e:
                    logger.error(f"Error building FEN from moves: {e}")
                    return None
            else:
                logger.warning("No FEN position or moves provided")
                return None
        
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            
            if not legal_moves:
                logger.warning("No legal moves available")
                return None
            
            selected_move = random.choice(legal_moves)
            logger.info(f"Selected random fallback move: {selected_move.uci()}")
            
            return selected_move.uci()
            
        except Exception as e:
            logger.error(f"Error generating fallback move: {e}", exc_info=True)
            return None
    
    def get_active_games(self) -> List[str]:
        """Get list of active game IDs.
        
        Returns:
            List of active game IDs
        """
        return [game_id for game_id, game in self.active_games.items() 
                if game['status'] == 'playing']
    
    def get_game_info(self, game_id: str) -> Optional[Dict]:
        """Get information about a specific game.
        
        Args:
            game_id: ID of the game
            
        Returns:
            Game information dict or None if game not found
        """
        return self.active_games.get(game_id)
    
    async def close_all_games(self):
        """Close all active game connections."""
        # Cancel all game monitoring tasks
        for game_id, task in self.game_tasks.items():
            try:
                task.cancel()
            except Exception as e:
                logger.error(f"Error canceling task for game {game_id}: {e}")
        
        self.game_tasks.clear()
        logger.info("Closed all game connections")
