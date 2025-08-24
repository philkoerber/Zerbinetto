![Zerbinetto Logo](logo.svg)

# Zerbinetto

A Lichess chess bot that plays automatically 24/7. Built with Python and the Lichess API, it handles game requests, makes legal moves, and manages its own game queue.

## Quick Start

```bash
# Build and run with Docker
make build
make prod

# Or run locally
pip install -r requirements.txt
python src/bot.py
```

## Commands

- `make build` - Build Docker image
- `make dev` - Run in development mode
- `make prod` - Run in production mode
- `make stop` - Stop all containers
- `make logs` - View logs
- `make status` - Check container status

## Setup

1. **Create Lichess Bot Account**
   - Go to https://lichess.org/account/oauth/token/create
   - Create a personal access token
   - Add token to `.env` file

2. **Configure Bot**
   - Copy `.env.example` to `.env`
   - Add your Lichess token
   - Set bot preferences (time controls, etc.)

3. **Deploy**
   - Use `make prod` for Docker deployment
   - Or run directly with `python src/bot.py`

## Features

- **Solid Chess Engine**: Clean, maintainable engine with fundamental chess principles
- **Automatic Play**: Accepts and plays games automatically
- **Legal Moves**: Uses python-chess for move validation
- **Queue Management**: Handles multiple game requests
- **Docker Support**: Containerized for easy deployment
- **Webhook Updates**: Automatic deployment from GitHub

## Project Structure

```
Zerbinetto/
├── src/                    # Main bot code
│   ├── bot.py             # Main bot script
│   ├── game_handler.py    # Game logic and move coordination
│   ├── lichess_client.py  # Lichess API client
│   ├── zerbinetto_engine.py # Solid chess engine
│   └── zerbinetto_config.py # Engine configuration
├── tests/                 # Test suite
│   └── test_engine.py     # Comprehensive engine tests
├── scripts/               # Deployment scripts
├── config/                # Configuration files
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker services
└── Makefile              # Build commands
```

## Chess Engine

Zerbinetto uses a clean, solid chess engine focused on fundamental principles:

- **Search Algorithm**: Minimax with alpha-beta pruning, move ordering, quiescence search
- **Positional Evaluation**: Piece-square tables, pawn structure, mobility
- **Playing Style**: Solid, principled chess
- **Strength**: Targets ~1800-2000 Elo rating
- **Performance**: 4 ply depth, ~1-3 seconds per move, transposition table caching
- **Configuration**: Adjustable parameters in `src/zerbinetto_config.py`

### Testing the Engine

Run the test suite to verify engine functionality:

```bash
# Activate virtual environment
source venv/bin/activate

# Run comprehensive engine tests
python tests/test_engine.py
```

## Configuration

The bot uses environment variables for configuration:

- `LICHESS_TOKEN` - Your Lichess API token
- `BOT_USERNAME` - Bot's username (optional)
- `LOG_LEVEL` - Logging level (INFO, DEBUG, etc.)

Engine parameters can be adjusted in `src/zerbinetto_config.py`:

## Troubleshooting

- **Bot not responding**: Check Lichess token in `.env`
- **Docker issues**: Use `make logs` to view container logs
- **API errors**: Verify token permissions on Lichess

## Architecture

- **Event-driven**: Responds to Lichess game events
- **Stateless**: No persistent storage required
- **Scalable**: Docker containers for easy scaling
- **Automated**: GitHub webhooks for continuous deployment
