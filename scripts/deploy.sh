#!/bin/bash

# Zerbinetto Deployment Script
# This script handles automatic deployment and updates

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/opt/zerbinetto"
DOCKER_COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
LOG_FILE="$PROJECT_DIR/deploy.log"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "$(date): [INFO] $1" >> "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "$(date): [SUCCESS] $1" >> "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "$(date): [WARNING] $1" >> "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "$(date): [ERROR] $1" >> "$LOG_FILE"
}

# Function to check if we're in the right directory
check_directory() {
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        print_error "Docker Compose file not found. Are you in the correct directory?"
        exit 1
    fi
}

# Function to stop and clean up existing containers
cleanup_docker() {
    print_status "Stopping and cleaning up existing containers..."
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        cd "$PROJECT_DIR"
        docker-compose down --remove-orphans || true
        docker system prune -f || true
        print_success "Docker cleanup completed"
    else
        print_warning "Docker Compose file not found, skipping cleanup"
    fi
}

# Function to pull latest changes from GitHub
pull_updates() {
    print_status "Pulling latest changes from GitHub..."
    
    cd "$PROJECT_DIR"
    
    # Check if we have uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_warning "Local changes detected, stashing them..."
        git stash
    fi
    
    # Pull latest changes
    git fetch origin
    git reset --hard origin/main
    
    print_success "Successfully pulled latest changes from GitHub"
}

# Function to build and start the application
build_and_start() {
    print_status "Building and starting the application..."
    
    cd "$PROJECT_DIR"
    
    # Build the Docker image
    docker-compose build --no-cache
    
    # Start the application
    docker-compose up -d
    
    print_success "Application built and started successfully"
}

# Function to start continuous training
start_training() {
    print_status "Starting continuous training..."
    
    cd "$PROJECT_DIR"
    
    # Start training in background inside the Docker container with logging
    docker exec -d zerbinetto-bot bash -c "python -m src.trainer --continuous --iterations 1000 --games-per-iteration 50 > /app/training.log 2>&1"
    
    print_success "Continuous training started in background with logging to training.log"
}

# Function to check application health
check_health() {
    print_status "Checking application health..."
    
    cd "$PROJECT_DIR"
    
    # Wait a moment for the application to start
    sleep 10
    
    # Check if the container is running
    if docker-compose ps | grep -q "Up"; then
        print_success "Application is running and healthy"
        return 0
    else
        print_error "Application failed to start properly"
        return 1
    fi
}

# Function to show deployment status
show_status() {
    print_status "Current deployment status:"
    cd "$PROJECT_DIR"
    docker-compose ps
}

# Function to rollback to previous version
rollback() {
    print_warning "Rolling back to previous version..."
    
    cd "$PROJECT_DIR"
    
    # Stop current containers
    docker-compose down
    
    # Reset to previous commit
    git reset --hard HEAD~1
    
    # Rebuild and start
    build_and_start
    
    print_success "Rollback completed"
}

# Function to show logs
show_logs() {
    print_status "Showing application logs..."
    cd "$PROJECT_DIR"
    docker-compose logs -f
}

# Function to show deployment log
show_deploy_log() {
    print_status "Showing deployment log..."
    tail -f "$LOG_FILE"
}

# Function to show training log
show_training_log() {
    print_status "Showing training log..."
    cd "$PROJECT_DIR"
    if [ -f "training.log" ]; then
        tail -f training.log
    else
        print_warning "Training log not found. Training may not be running."
    fi
}

# Function to check training status
check_training_status() {
    print_status "Checking training status..."
    cd "$PROJECT_DIR"
    
    # Check if training process is running
    if docker exec zerbinetto-bot ps aux | grep -q "trainer"; then
        print_success "Training process is running"
    else
        print_warning "Training process is not running"
    fi
    
    # Show recent training log entries
    if [ -f "training.log" ]; then
        print_status "Recent training log entries:"
        tail -10 training.log
    else
        print_warning "No training log found"
    fi
}

# Function to show help
show_help() {
    echo "Zerbinetto Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy      Full deployment (pull, cleanup, build, start, train)"
    echo "  update      Pull updates and restart (triggered by webhook)"
    echo "  restart     Restart the application"
    echo "  train       Start continuous training"
    echo "  stop        Stop the application"
    echo "  status      Show application status"
    echo "  logs        Show application logs"
    echo "  deploy-log  Show deployment log"
    echo "  training-log Show training log"
    echo "  training-status Check training status"
    echo "  rollback    Rollback to previous version"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy   # Full deployment"
    echo "  $0 update   # Update and restart (webhook-triggered)"
    echo "  $0 logs     # View logs"
    echo "  $0 training-status # Check training progress"
}

# Main deployment function
deploy() {
    print_status "Starting full deployment..."
    
    check_directory
    cleanup_docker
    pull_updates
    build_and_start
    
    if check_health; then
        print_success "Deployment completed successfully!"
        start_training
        show_status
    else
        print_error "Deployment failed!"
        rollback
        exit 1
    fi
}

# Update function (pull and restart)
update() {
    print_status "Starting update..."
    
    check_directory
    pull_updates
    cleanup_docker
    build_and_start
    
    if check_health; then
        print_success "Update completed successfully!"
        show_status
    else
        print_error "Update failed!"
        rollback
        exit 1
    fi
}

# Main script logic
main() {
    # Create log file if it doesn't exist
    touch "$LOG_FILE"
    
    case "${1:-help}" in
        "deploy")
            deploy
            ;;
        "update")
            update
            ;;
        "restart")
            check_directory
            cd "$PROJECT_DIR"
            docker-compose restart
            print_success "Application restarted"
            ;;
        "train")
            start_training
            ;;
        "stop")
            check_directory
            cd "$PROJECT_DIR"
            docker-compose down
            print_success "Application stopped"
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "deploy-log")
            show_deploy_log
            ;;
        "training-log")
            show_training_log
            ;;
        "training-status")
            check_training_status
            ;;
        "rollback")
            rollback
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
