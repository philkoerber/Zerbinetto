#!/usr/bin/env python3
"""
Lichess Bot

A minimal bot that accepts challenges and plays random legal moves.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Optional

from lichess_client import LichessClient
from game_handler import GameHandler

# Configuration
LOG_LEVEL = logging.INFO
CHALLENGE_TIMEOUT = 30  # seconds to wait for opponent to move
MOVE_TIMEOUT = 5  # seconds to wait before making a move

# Global variables for cleanup
bot_client: Optional[LichessClient] = None
game_handler: Optional[GameHandler] = None

def setup_logging(debug: bool = False):
    """Set up logging configuration.
    
    Args:
        debug: Enable debug logging if True
    """
    level = logging.DEBUG if debug else LOG_LEVEL
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

async def handle_challenge(challenge_data: dict):
    """Handle incoming challenges.
    
    Args:
        challenge_data: Challenge event data from Lichess
    """
    challenge = challenge_data.get('challenge', {})
    challenge_id = challenge.get('id')
    challenger = challenge.get('challenger', {}).get('name', 'Unknown')
    
    if not challenge_id:
        logger.warning("Challenge event missing challenge ID")
        return
    
    logger.info(f"Received challenge from {challenger} (ID: {challenge_id})")
    
    # Accept all challenges
    try:
        await bot_client.accept_challenge(challenge_id)
        logger.info(f"Accepted challenge from {challenger}")
    except Exception as e:
        logger.error(f"Failed to accept challenge from {challenger}: {e}")

async def handle_game_start(game_data: dict):
    """Handle game start events.
    
    Args:
        game_data: Game start event data
    """
    await game_handler.handle_game_start(game_data)

async def handle_game_finish(game_data: dict):
    """Handle game finish events.
    
    Args:
        game_data: Game finish event data
    """
    await game_handler.handle_game_finish(game_data)

async def handle_game_full(game_data: dict):
    """Handle game full events (initial game state).
    
    Args:
        game_data: Game full event data
    """
    await game_handler.handle_game_state(game_data)

async def handle_game_state(game_data: dict):
    """Handle game state updates.
    
    Args:
        game_data: Game state event data
    """
    await game_handler.handle_game_state(game_data)

async def main_loop():
    """Main bot loop."""
    global bot_client, game_handler
    
    try:
        # Initialize client and game handler
        bot_client = LichessClient()
        game_handler = GameHandler(bot_client, zerb_style=args.zerb_style)
        
        # Get account info
        account_info = bot_client.get_account_info()
        if account_info:
            username = account_info.get('username', 'Unknown')
            logger.info(f"Logged in as: {username}")
        else:
            logger.warning("Could not retrieve account info")
        
        # Set up event handlers
        bot_client.add_event_handler('challenge', handle_challenge)
        bot_client.add_event_handler('gameStart', handle_game_start)
        bot_client.add_event_handler('gameFinish', handle_game_finish)
        bot_client.add_event_handler('gameFull', handle_game_full)
        bot_client.add_event_handler('gameState', handle_game_state)
        
        # Connect to Lichess
        await bot_client.connect()
        
        logger.info("Bot is running! Press Ctrl+C to stop.")
        logger.info("Waiting for challenges...")
        
        # Listen for events
        await bot_client.listen_for_events()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        await cleanup()

async def cleanup():
    """Clean up resources."""
    global bot_client, game_handler
    
    logger.info("Cleaning up...")
    
    if game_handler:
        await game_handler.close_all_games()
    
    if bot_client:
        await bot_client.disconnect()
    
    logger.info("Cleanup complete")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Lichess Bot - Accepts challenges and plays random moves')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--zerb-style', action='store_true', help='Play in spectacular sacrificial style')
    parser.add_argument('--token', help='Lichess API token (overrides LICHESS_TOKEN env var)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check for API token
    if not args.token and not os.getenv('LICHESS_TOKEN'):
        logger.error("Lichess API token is required!")
        logger.error("Set LICHESS_TOKEN environment variable or use --token argument")
        logger.error("Get your token from: https://lichess.org/account/oauth/token")
        sys.exit(1)
    
    # Set token if provided via command line
    if args.token:
        os.environ['LICHESS_TOKEN'] = args.token
    
    # Run the bot
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()
