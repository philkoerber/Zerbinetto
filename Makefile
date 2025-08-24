.PHONY: help build dev prod stop logs test test-ml clean

# Default target
help:
	@echo "Zerbinetto Lichess Bot - Available Commands:"
	@echo ""
	@echo "  make build    - Build Docker image"
	@echo "  make dev      - Run bot in development mode"
	@echo "  make prod     - Run bot in production mode"
	@echo "  make stop     - Stop the bot"
	@echo "  make logs     - View bot logs"
	@echo "  make test     - Test bot setup"
	@echo "  make test-ml  - Test ML engine"
	@echo "  make clean    - Clean up Docker resources"
	@echo "  make help     - Show this help message"

# Build Docker image
build:
	@echo "Building Docker image..."
	docker-compose build

# Run in development mode
dev: build
	@echo "Starting bot in development mode..."
	@mkdir -p logs
	docker-compose --profile dev up lichess-bot-dev

# Run in production mode
prod: build
	@echo "Starting bot in production mode..."
	@mkdir -p logs
	docker-compose up -d lichess-bot
	@echo "Bot started in background. Check logs with: make logs"

# Stop the bot
stop:
	@echo "Stopping bot..."
	docker-compose down

# View logs
logs:
	@echo "Showing bot logs..."
	docker-compose logs -f lichess-bot

# Test setup
test: build
	@echo "Testing bot setup..."
	docker-compose run --rm lichess-bot python -m pytest tests/ -v

# Test ML engine
test-ml: build
	@echo "Testing ML engine..."
	docker-compose run --rm lichess-bot python tests/test_ml_engine.py

# Clean up Docker resources
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

# Show status
status:
	@echo "Bot status:"
	docker-compose ps
