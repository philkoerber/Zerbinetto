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
import websockets

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
        self.game_websockets: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Configuration
        self.move_delay = 1.0  # Delay before making a move (seconds)
        self.max_move_time = 10.0  # Maximum time to think about a move (seconds)
    
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
        
        # Store game info
        self.active_games[game_id] = {
            'id': game_id,
            'started_at': time.time(),
            'moves': [],
            'status': 'playing'
        }
        
        # Connect to game WebSocket
        await self._connect_to_game(game_id)
    
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
        
        # Close game WebSocket
        if game_id in self.game_websockets:
            await self.game_websockets[game_id].close()
            del self.game_websockets[game_id]
    
    async def handle_game_state(self, game_data: Dict):
        """Handle game state updates.
        
        Args:
            game_data: Game state event data
        """
        game_id = game_data.get('id')
        if not game_id:
            logger.warning("Game state event missing game ID")
            return
        
        if game_id not in self.active_games:
            logger.warning(f"Received game state for unknown game: {game_id}")
            return
        
        # Update game state
        game = self.active_games[game_id]
        game['moves'] = game_data.get('moves', '').split()
        game['status'] = game_data.get('status')
        
        # Check if it's our turn
        if self._is_our_turn(game_data):
            await self._make_move(game_id, game_data)
    
    async def _connect_to_game(self, game_id: str):
        """Connect to a specific game's WebSocket stream.
        
        Args:
            game_id: ID of the game to connect to
        """
        try:
            ws_url = f"wss://lichess.org/api/board/game/stream/{game_id}"
            websocket = await websockets.connect(
                ws_url,
                extra_headers={"Authorization": f"Bearer {self.lichess_client.token}"}
            )
            
            self.game_websockets[game_id] = websocket
            
            # Start listening for game events
            asyncio.create_task(self._listen_to_game(game_id, websocket))
            
            logger.info(f"Connected to game WebSocket: {game_id}")
        except Exception as e:
            logger.error(f"Failed to connect to game {game_id}: {e}")
    
    async def _listen_to_game(self, game_id: str, websocket):
        """Listen for events from a specific game.
        
        Args:
            game_id: ID of the game
            websocket: WebSocket connection to the game
        """
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_game_event(game_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from game {game_id}: {message}")
                except Exception as e:
                    logger.error(f"Error handling game event for {game_id}: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Game WebSocket closed for {game_id}")
        except Exception as e:
            logger.error(f"Game WebSocket error for {game_id}: {e}")
        finally:
            if game_id in self.game_websockets:
                del self.game_websockets[game_id]
    
    async def _handle_game_event(self, game_id: str, data: Dict):
        """Handle events from a specific game.
        
        Args:
            game_id: ID of the game
            data: Event data
        """
        if 'type' in data:
            if data['type'] == 'gameFull':
                # Initial game state
                await self.handle_game_state(data)
            elif data['type'] == 'gameState':
                # Game state update
                await self.handle_game_state(data)
            elif data['type'] == 'chatLine':
                # Chat message (ignore for now)
                pass
            else:
                logger.debug(f"Unhandled game event type: {data['type']}")
        else:
            # Assume it's a game state update
            await self.handle_game_state(data)
    
    def _is_our_turn(self, game_data: Dict) -> bool:
        """Check if it's our turn to move.
        
        Args:
            game_data: Game state data
            
        Returns:
            True if it's our turn, False otherwise
        """
        # This is a simplified check - in a real implementation,
        # you'd need to parse the FEN and determine whose turn it is
        # For now, we'll assume if there are moves and the game is playing,
        # it might be our turn
        moves = game_data.get('moves', '')
        status = game_data.get('status')
        
        # If game is still playing and there are moves, it might be our turn
        # This is a very basic heuristic - a real implementation would need
        # proper chess position analysis
        return status == 'started' and len(moves.split()) > 0
    
    async def _make_move(self, game_id: str, game_data: Dict):
        """Make a move in the game.
        
        Args:
            game_id: ID of the game
            game_data: Current game state
        """
        try:
            # Add a small delay to avoid making moves too quickly
            await asyncio.sleep(self.move_delay)
            
            # Generate a random legal move
            move = await self._generate_random_move(game_data)
            
            if move:
                await self.lichess_client.make_move(game_id, move)
                logger.info(f"Made move {move} in game {game_id}")
            else:
                logger.warning(f"No legal moves available in game {game_id}")
                
        except Exception as e:
            logger.error(f"Error making move in game {game_id}: {e}")
    
    async def _generate_random_move(self, game_data: Dict) -> Optional[str]:
        """Generate a random legal move.
        
        Args:
            game_data: Current game state
            
        Returns:
            A legal move in UCI format, or None if no moves available
        """
        # This is a simplified implementation that generates random moves
        # In a real chess bot, you would:
        # 1. Parse the FEN string to get the current position
        # 2. Generate all legal moves for the current position
        # 3. Choose one randomly or using some strategy
        
        # For now, we'll generate some common opening moves
        # This is NOT a proper chess engine - just for demonstration
        common_moves = [
            "e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "f2f4", "e2e3", "d2d3",
            "g2g3", "b2b3", "c2c3", "f2f3", "h2h3", "a2a3", "g2g4", "h2h4"
        ]
        
        # Add some pawn moves
        for file in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            common_moves.append(f"{file}2{file}4")
            common_moves.append(f"{file}2{file}3")
        
        # Add some knight moves
        knight_moves = ["b1a3", "b1c3", "g1f3", "g1h3"]
        common_moves.extend(knight_moves)
        
        # Randomly select a move
        return random.choice(common_moves)
    
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
        for game_id, websocket in self.game_websockets.items():
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing game {game_id}: {e}")
        
        self.game_websockets.clear()
        logger.info("Closed all game connections")
