<div align="center">
  <img src="logo.svg" alt="Zerbinetto Bot Logo" width="200"/>
  
  # Zerbinetto - Lichess Chess Bot
  
  *A fully functional Lichess chess bot that accepts challenges and plays legal moves using proper chess logic.*
</div>

## ✨ Current Status: **FULLY WORKING** ✨

**The bot is now completely functional and ready to play chess!** 🎉

### What Works:
- ✅ **Challenge Acceptance**: Automatically accepts all incoming challenges
- ✅ **Legal Move Generation**: Uses `python-chess` library for proper chess logic  
- ✅ **Complete Games**: Plays from opening to endgame
- ✅ **Both Colors**: Works correctly as white or black
- ✅ **Real-time Streaming**: Monitors games via Lichess API streams
- ✅ **Turn Detection**: Accurately detects when it's the bot's turn to move
- ✅ **Error Recovery**: Handles network issues and API errors gracefully
- ✅ **Bot Flair**: Full support for BOT account upgrade

### Recent Improvements:
- 🔧 **Fixed Move Generation**: Now generates only legal moves in any position
- 🔧 **Fixed Game Streaming**: Proper real-time game state monitoring  
- 🔧 **Fixed Turn Logic**: Accurate detection of whose turn it is
- 🔧 **Added Chess Engine**: Integrated python-chess for proper game logic
- 🔧 **Enhanced Logging**: Comprehensive debugging and monitoring

## Features

- **Proper Chess Logic**: Uses the `python-chess` library for legal move generation
- **Real-time Game Streaming**: Monitors games via Lichess API streams
- **Robust Move Generation**: Only generates and attempts legal moves
- **Automatic Challenge Acceptance**: Accepts all incoming challenges
- **Complete Game Flow**: Handles games from start to finish
- **Bot Account Support**: Full support for BOT flair accounts
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Error Recovery**: Fallback mechanisms for network issues

## Prerequisites

- Python 3.8 or higher
- A Lichess account with API token
- **For Bot Flair**: A separate bot account (see Bot Setup section below)

## Setup

### 1. Get a Lichess API Token

#### Option A: Regular Account (No Bot Flair)
1. Go to [Lichess Settings](https://lichess.org/account/oauth/token)
2. Create a new personal API access token
3. Copy the token (you'll need it for the next step)

#### Option B: Bot Account (With Bot Flair) - **Recommended**
1. Create a new Lichess account (don't play any games on it)
2. Go to [lichess.org/api](https://lichess.org/api) → BOT → "Upgrade BOT account"
3. Create a personal API token with all permissions checked
4. Copy the token and add it to your `.env` file
5. Run the bot upgrade script:
   ```bash
   # Using Python script
   python upgrade_to_bot.py
   
   # Or using shell script
   ./upgrade_to_bot.sh
   ```
6. Your account will now have "BOT" in front of the username

### 2. Install Dependencies

#### Option 1: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using pipx (Alternative)

```bash
# Install pipx if not already installed
# On macOS: brew install pipx
# On Ubuntu: sudo apt install pipx

# Install dependencies
pipx install requests python-dotenv websockets
```

#### Option 3: System-wide Installation (Not Recommended)

```bash
pip install --user -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
LICHESS_TOKEN=your_api_token_here
```

Replace `your_api_token_here` with your actual Lichess API token.

## Usage

### Local Development

#### Quick Start (Recommended)

Use the provided startup script:

```bash
./start_bot.sh
```

This script will:
- Create and activate a virtual environment if needed
- Install dependencies automatically
- Start the bot with proper configuration

#### Manual Start

If you prefer to run manually:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Start the bot
python bot.py
```

The bot will:
- Connect to Lichess event stream
- Accept all incoming challenges automatically  
- Play legal chess moves using proper game logic
- Monitor active games in real-time
- Log detailed activities to the console

### Production Deployment

#### Option 1: Using systemd (Linux)

1. Copy the bot to your server
2. Install dependencies: `pip install -r requirements.txt`
3. Set up the environment file with your token
4. Copy the systemd service file:

```bash
sudo cp lichess-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lichess-bot
sudo systemctl start lichess-bot
```

Check status: `sudo systemctl status lichess-bot`

#### Option 2: Using screen/tmux

```bash
# Start a new screen session
screen -S lichess-bot

# Run the bot
python bot.py

# Detach from screen (Ctrl+A, then D)
# To reattach: screen -r lichess-bot
```

#### Option 3: Using nohup

```bash
nohup python bot.py > bot.log 2>&1 &
```

## Bot Setup (Getting Bot Flair)

### Quick Bot Upgrade

To get the bot flair (BOT prefix on your username), use the included upgrade scripts:

```bash
# Method 1: Python script (recommended)
python upgrade_to_bot.py

# Method 2: Shell script
./upgrade_to_bot.sh

# Verify bot status
python upgrade_to_bot.py verify
```

### Manual Bot Upgrade

If you prefer to do it manually:

1. Create a new Lichess account (don't play any games)
2. Go to [lichess.org/api](https://lichess.org/api) → BOT → "Upgrade BOT account"
3. Create a personal API token with all permissions checked
4. Add the token to your `.env` file
5. Run this curl command:
   ```bash
   curl -d '' lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YOUR_TOKEN_HERE"
   ```

### Bot Account Best Practices

- **Separate Account**: Use a different account for your bot (not your main account)
- **No Games**: Don't play any games on the bot account before upgrading
- **Clear Username**: Choose a username that indicates it's a bot
- **Proper Identification**: The bot code includes proper User-Agent headers

## Project Structure

```
lichess-bot/
├── bot.py              # Main bot script and event loop
├── lichess_client.py   # Lichess API client (REST + streaming)
├── game_handler.py     # Chess game logic and legal move generation
├── upgrade_to_bot.py   # Bot account upgrade script
├── upgrade_to_bot.sh   # Bot upgrade shell script
├── requirements.txt    # Python dependencies (includes python-chess)
├── env.example         # Environment variables template
├── start_bot.sh        # Quick start script
├── deploy.sh           # Deployment script
├── test_setup.py       # Setup verification script
├── lichess-bot.service # systemd service file
├── .gitignore          # Git ignore file
└── README.md          # This file
```

## Configuration

The bot can be configured by modifying the constants in `bot.py`:

- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `CHALLENGE_TIMEOUT`: Time to wait for opponent to move (seconds)
- `MOVE_TIMEOUT`: Time to wait before making a move (seconds)

## Logging

The bot provides comprehensive logging including:
- **Connection Status**: Stream connections and disconnections
- **Challenge Handling**: Challenges received, accepted, and declined
- **Game Events**: Game starts, finishes, and state updates
- **Move Generation**: Legal moves found and selected
- **Chess Logic**: FEN positions, move validation, and board states
- **Error Handling**: Network issues, API errors, and recovery attempts
- **Performance**: Move timing and game monitoring status

Logs are written to the console by default. For production, consider redirecting to a log file.

## Troubleshooting

### Common Issues

1. **Connection failed**: Check your internet connection and API token
2. **Token invalid**: Verify your Lichess API token is correct
3. **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)

### Debug Mode

Run with debug logging:

```bash
# Using startup script
./start_bot.sh --debug

# Or manually
python bot.py --debug
```

## Development

### Adding Features

The bot is designed to be easily extensible:

- **`lichess_client.py`**: Handles all Lichess API communication (REST + streaming)
- **`game_handler.py`**: Implements chess game logic and legal move generation using python-chess
- **`bot.py`**: Main bot event loop and coordination between components

### Current Architecture

- **Event-Driven**: Uses Lichess event streams for real-time updates
- **Asynchronous**: Built with asyncio for concurrent game handling  
- **Chess Engine**: Integrates python-chess library for proper game logic
- **Robust**: Includes error handling and fallback mechanisms
- **Modular**: Clean separation between API, game logic, and coordination

### Testing

#### Verify Setup

Before running the bot, verify your setup:

```bash
python test_setup.py
```

This will check:
- Python version compatibility
- All required dependencies
- Bot module imports
- Environment configuration

#### Test the Bot

The bot has been thoroughly tested and is working correctly:

**✅ Verified Working Features:**
- Challenge acceptance from any user
- Legal move generation in all positions
- Complete game flow (start to finish)  
- Both white and black piece handling
- Real-time game state monitoring
- Proper turn detection and move timing

**For your own testing:**
1. Create a test Lichess account
2. Use a different API token  
3. Challenge the bot from your main account
4. The bot will accept and play a complete game

## License

This project is open source. Feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Disclaimer

This bot is for educational purposes. Please respect Lichess terms of service and fair play policies.
