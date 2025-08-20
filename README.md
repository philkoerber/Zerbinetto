# Lichess Bot

A minimal Lichess bot that accepts challenges and plays random legal moves.

## Features

- Connects to Lichess using API token authentication
- Automatically accepts all incoming challenges
- Plays random legal moves until the game ends
- Logs basic information (challenges, moves, game results)
- Simple start/stop functionality

## Prerequisites

- Python 3.8 or higher
- A Lichess account with API token

## Setup

### 1. Get a Lichess API Token

1. Go to [Lichess Settings](https://lichess.org/account/oauth/token)
2. Create a new personal API access token
3. Copy the token (you'll need it for the next step)

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
- Connect to Lichess
- Accept all incoming challenges
- Play random legal moves
- Log activities to the console

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

## Project Structure

```
lichess-bot/
├── bot.py              # Main bot script
├── lichess_client.py   # Lichess API client
├── game_handler.py     # Game logic and move generation
├── requirements.txt    # Python dependencies
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

The bot logs the following information:
- Connection status
- Challenges received and accepted
- Games started/ended
- Moves made
- Errors and exceptions

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

- `lichess_client.py`: Handle Lichess API communication
- `game_handler.py`: Implement chess logic and move generation
- `bot.py`: Main bot loop and coordination

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

For testing, you can:
1. Create a test Lichess account
2. Use a different API token
3. Challenge the bot from your main account

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
