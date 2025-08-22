<div align="center">
  <img src="logo.svg" alt="Zerbinetto Bot Logo" width="200"/>
  
  # Zerbinetto - Lichess Chess Bot
  
  *A fully functional Lichess chess bot that accepts challenges and plays legal moves.*
</div>

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Lichess API token

### Setup

1. **Get a Lichess API Token**
   ```bash
   # Go to: https://lichess.org/account/oauth/token
   # Create a new personal API access token
   ```

2. **Configure Environment**
   ```bash
   cp env.example .env
   # Edit .env and add your LICHESS_TOKEN
   ```

3. **Run the Bot**
   ```bash
   # Development mode (with debug logging)
   make dev
   
   # Production mode (background)
   make prod
   ```

## Commands

```bash
make dev      # Run in development mode
make prod     # Run in production mode
make stop     # Stop the bot
make logs     # View logs
make test     # Test setup
make upgrade  # Upgrade bot account
make help     # Show all commands
```

## Project Structure

```
Zerbinetto/
├── src/                    # Source code
│   ├── bot.py             # Main bot script
│   ├── lichess_client.py  # Lichess API client
│   └── game_handler.py    # Chess game logic
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── Makefile               # Build automation
└── requirements.txt       # Python dependencies
```

## Bot Account Setup (Optional)

To get "BOT" flair on your username:

1. Create a new Lichess account (don't play games)
2. Get API token from https://lichess.org/account/oauth/token
3. Update .env with the new token
4. Run: `make upgrade`

## Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run bot
python src/bot.py

# Run with debug logging
python src/bot.py --debug
```

## Features

- ✅ Accepts all incoming challenges
- ✅ Plays legal chess moves using python-chess
- ✅ Real-time game streaming
- ✅ Complete game flow (start to finish)
- ✅ Error recovery and retry logic
- ✅ Docker containerization for easy deployment

## Configuration

Edit environment variables in `.env`:
- `LICHESS_TOKEN`: Your Lichess API token

## Troubleshooting

- **Docker not running**: Start Docker Desktop
- **Token invalid**: Check your Lichess API token
- **Import errors**: Run `make build` to rebuild image

## Architecture

- **Event-driven**: Uses Lichess event streams for real-time updates
- **Asynchronous**: Built with asyncio for concurrent game handling
- **Chess Engine**: Integrates python-chess library for proper game logic
- **Containerized**: Docker-based deployment for consistency
- **Stateless**: No persistent data storage required

## License

Open source - feel free to modify and distribute.
