#!/bin/bash

# Docker Runner Script for Zerbinetto Lichess Bot
# This script makes it easy to run the bot in different modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if .env file exists
check_env() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from env.example..."
        if [ -f "env.example" ]; then
            cp env.example .env
            print_warning "Please edit .env file and add your LICHESS_TOKEN"
        else
            print_error "No .env file or env.example found. Please create .env with LICHESS_TOKEN"
            exit 1
        fi
    fi
    
    if ! grep -q "LICHESS_TOKEN=" .env || grep -q "LICHESS_TOKEN=your_api_token_here" .env; then
        print_error "LICHESS_TOKEN not set in .env file. Please add your token."
        exit 1
    fi
}

# Function to create logs directory
create_logs_dir() {
    if [ ! -d "logs" ]; then
        mkdir -p logs
        print_status "Created logs directory"
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
    print_success "Docker image built successfully"
}

# Function to run in development mode
run_dev() {
    print_status "Starting bot in development mode..."
    create_logs_dir
    docker-compose --profile dev up lichess-bot-dev
}

# Function to run in production mode
run_prod() {
    print_status "Starting bot in production mode..."
    create_logs_dir
    docker-compose up -d lichess-bot
    print_success "Bot started in production mode (background)"
    print_status "Check logs with: docker-compose logs -f lichess-bot"
}

# Function to stop the bot
stop_bot() {
    print_status "Stopping bot..."
    docker-compose down
    print_success "Bot stopped"
}

# Function to view logs
view_logs() {
    print_status "Showing bot logs..."
    docker-compose logs -f lichess-bot
}

# Function to test the setup
test_setup() {
    print_status "Testing bot setup..."
    docker-compose run --rm lichess-bot python -m pytest tests/ -v
}

# Function to test ML engine
test_ml() {
    print_status "Testing ML engine..."
    docker-compose run --rm lichess-bot python tests/test_ml_engine.py
}

# Function to show status
show_status() {
    print_status "Bot status:"
    docker-compose ps
}

# Function to show help
show_help() {
    echo "Zerbinetto Lichess Bot - Docker Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev         Run bot in development mode (with debug logging)"
    echo "  prod        Run bot in production mode (background)"
    echo "  stop        Stop the bot"
    echo "  logs        View bot logs"
    echo "  build       Build Docker image"
    echo "  test        Test bot setup"
    echo "  test-ml     Test ML engine"
    echo "  status      Show bot status"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev      # Run in development mode"
    echo "  $0 prod     # Run in production mode"
    echo "  $0 logs     # View logs"
}

# Main script logic
main() {
    check_docker
    check_env
    
    case "${1:-help}" in
        "dev")
            build_image
            run_dev
            ;;
        "prod")
            build_image
            run_prod
            ;;
        "stop")
            stop_bot
            ;;
        "logs")
            view_logs
            ;;
        "build")
            build_image
            ;;
        "test")
            build_image
            test_setup
            ;;
        "test-ml")
            build_image
            test_ml
            ;;
        "status")
            show_status
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
