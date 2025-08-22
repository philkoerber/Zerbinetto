#!/bin/bash

# Lichess Bot Startup Script
# This script activates the virtual environment and starts the bot

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    # Activate virtual environment
    source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found!"
    echo "Please create a .env file with your Lichess API token:"
    echo "  cp env.example .env"
    echo "  # Then edit .env and add your token"
    echo ""
    echo "You can also run the bot with: python bot.py --token YOUR_TOKEN"
    echo ""
fi

# Start the bot
echo "Starting Lichess Bot..."
python bot.py "$@"
