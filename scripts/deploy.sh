#!/bin/bash

# Lichess Bot Deployment Script
# This script helps set up the bot on a Linux server

set -e

echo "=== Lichess Bot Deployment Script ==="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root. Please run as a regular user."
   exit 1
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Python $python_version is installed, but Python $required_version or higher is required."
    exit 1
fi

echo "Python $python_version detected - OK"

# Create bot directory
BOT_DIR="$HOME/lichess-bot"
echo "Setting up bot in: $BOT_DIR"

if [ ! -d "$BOT_DIR" ]; then
    mkdir -p "$BOT_DIR"
fi

# Copy files to bot directory
echo "Copying bot files..."
cp bot.py lichess_client.py game_handler.py requirements.txt "$BOT_DIR/"

# Install dependencies
echo "Installing Python dependencies..."
cd "$BOT_DIR"
python3 -m pip install --user -r requirements.txt

# Set up environment file
if [ ! -f "$BOT_DIR/.env" ]; then
    echo "Creating .env file..."
    cp env.example "$BOT_DIR/.env"
    echo ""
    echo "IMPORTANT: Please edit $BOT_DIR/.env and add your Lichess API token!"
    echo "Get your token from: https://lichess.org/account/oauth/token"
    echo ""
else
    echo ".env file already exists"
fi

# Make bot executable
chmod +x "$BOT_DIR/bot.py"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit $BOT_DIR/.env and add your Lichess API token"
echo "2. Test the bot: cd $BOT_DIR && python3 bot.py"
echo "3. For production deployment, see README.md for systemd setup"
echo ""
echo "To run the bot:"
echo "  cd $BOT_DIR"
echo "  python3 bot.py"
echo ""
echo "To run with debug logging:"
echo "  python3 bot.py --debug"
