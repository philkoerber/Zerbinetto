"""
Lichess API Client

Handles authentication, WebSocket connection, and API calls to Lichess.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Callable
import websockets
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LichessClient:
    """Client for interacting with Lichess API and WebSocket."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the Lichess client.
        
        Args:
            token: Lichess API token. If None, will try to load from LICHESS_TOKEN env var.
        """
        self.token = token or os.getenv('LICHESS_TOKEN')
        if not self.token:
            raise ValueError("Lichess API token is required. Set LICHESS_TOKEN environment variable or pass token parameter.")
        
        self.base_url = "https://lichess.org"
        self.ws_url = "wss://lichess.org/api/stream/event"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        self.websocket = None
        self.is_connected = False
        self.event_handlers: Dict[str, List[Callable]] = {
            'challenge': [],
            'gameStart': [],
            'gameFinish': [],
            'gameFull': [],
            'gameState': []
        }
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler for a specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event occurs
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def connect(self):
        """Connect to Lichess WebSocket stream."""
        try:
            logger.info("Connecting to Lichess WebSocket...")
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers={"Authorization": f"Bearer {self.token}"}
            )
            self.is_connected = True
            logger.info("Connected to Lichess WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to Lichess: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Lichess WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from Lichess WebSocket")
    
    async def listen_for_events(self):
        """Listen for events from Lichess WebSocket."""
        if not self.websocket:
            raise RuntimeError("Not connected to Lichess WebSocket")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_event(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error handling event: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
    
    async def _handle_event(self, data: Dict):
        """Handle incoming events from Lichess.
        
        Args:
            data: Event data from Lichess
        """
        event_type = data.get('type')
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
        else:
            logger.debug(f"Unhandled event type: {event_type}")
    
    async def accept_challenge(self, challenge_id: str):
        """Accept a challenge.
        
        Args:
            challenge_id: ID of the challenge to accept
        """
        url = f"{self.base_url}/api/challenge/{challenge_id}/accept"
        try:
            response = requests.post(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Accepted challenge: {challenge_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to accept challenge {challenge_id}: {e}")
    
    async def decline_challenge(self, challenge_id: str, reason: str = "generic"):
        """Decline a challenge.
        
        Args:
            challenge_id: ID of the challenge to decline
            reason: Reason for declining (generic, later, tooFast, tooSlow, timeControl, rated, casual, standard, variant, noBot, onlyBot)
        """
        url = f"{self.base_url}/api/challenge/{challenge_id}/decline"
        try:
            response = requests.post(url, headers=self.headers, json={"reason": reason})
            response.raise_for_status()
            logger.info(f"Declined challenge: {challenge_id} (reason: {reason})")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to decline challenge {challenge_id}: {e}")
    
    async def make_move(self, game_id: str, move: str):
        """Make a move in a game.
        
        Args:
            game_id: ID of the game
            move: Move in UCI format (e.g., "e2e4")
        """
        url = f"{self.base_url}/api/board/game/{game_id}/move/{move}"
        try:
            response = requests.post(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Made move {move} in game {game_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to make move {move} in game {game_id}: {e}")
    
    async def abort_game(self, game_id: str):
        """Abort a game.
        
        Args:
            game_id: ID of the game to abort
        """
        url = f"{self.base_url}/api/board/game/{game_id}/abort"
        try:
            response = requests.post(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Aborted game: {game_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to abort game {game_id}: {e}")
    
    async def resign_game(self, game_id: str):
        """Resign from a game.
        
        Args:
            game_id: ID of the game to resign from
        """
        url = f"{self.base_url}/api/board/game/{game_id}/resign"
        try:
            response = requests.post(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Resigned from game: {game_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to resign from game {game_id}: {e}")
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information.
        
        Returns:
            Account information dict or None if failed
        """
        url = f"{self.base_url}/api/account"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get account info: {e}")
            return None
