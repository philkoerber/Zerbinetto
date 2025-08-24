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
    
    # Fix model permissions after container starts
    fix_model_permissions
    
    print_success "Application built and started successfully"
}

# Function to fix model permissions
fix_model_permissions() {
    print_status "Fixing model file permissions..."
    
    cd "$PROJECT_DIR"
    
    # Fix permissions for the model file so the bot can write to it
    sudo chown 1000:1000 models/chess_model.pkl 2>/dev/null || true
    sudo chmod 644 models/chess_model.pkl 2>/dev/null || true
    
    print_success "Model permissions fixed"
}

# Function to start continuous training
start_training() {
    print_status "Starting continuous training..."
    
    cd "$PROJECT_DIR"
    
    # Fix permissions first
    fix_model_permissions
    
    # Start training in background inside the Docker container with log rotation
    # Run training indefinitely with log rotation to prevent disk space issues
    docker exec -d zerbinetto-bot bash -c "
        # Create log rotation script
        cat > /app/rotate_logs.sh << 'EOF'
        #!/bin/bash
        while true; do
            sleep 3600  # Check every hour
            if [ -f /app/training.log ] && [ \$(stat -c%s /app/training.log) -gt 10485760 ]; then  # 10MB
                mv /app/training.log /app/training.log.old
                touch /app/training.log
                # Keep only last 5 log files
                ls -t /app/training.log.* | tail -n +6 | xargs -r rm
            fi
        done
        EOF
        chmod +x /app/rotate_logs.sh
        
        # Start log rotation in background
        /app/rotate_logs.sh &
        
        # Start training with log rotation
        python -m src.trainer --continuous --forever --games-per-iteration 50 > /app/training.log 2>&1
    "
    
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

# Function to cleanup container and logs
cleanup_container() {
    print_status "Cleaning up container and logs..."
    
    cd "$PROJECT_DIR"
    
    # Stop container
    docker-compose down
    
    # Clean up old logs and temporary files
    docker system prune -f
    
    # Clean up old training logs (keep only last 5 files)
    if [ -d "logs" ]; then
        find logs/ -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
    fi
    
    # Clean up old model backups (keep only last 3)
    if [ -d "models" ]; then
        ls -t models/chess_model.pkl.backup_* 2>/dev/null | tail -n +4 | xargs -r rm
    fi
    
    print_success "Cleanup completed"
}

# Function to check disk usage
check_disk_usage() {
    print_status "Checking disk usage..."
    
    cd "$PROJECT_DIR"
    
    # Check overall disk usage
    disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
    print_status "Disk usage: ${disk_usage}%"
    
    if [ "$disk_usage" -gt 80 ]; then
        print_warning "High disk usage detected! Consider running cleanup."
    fi
    
    # Check Docker disk usage
    docker_usage=$(docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}" 2>/dev/null || echo "Docker not available")
    print_status "Docker disk usage:"
    echo "$docker_usage"
    
    # Check model file sizes
    if [ -d "models" ]; then
        print_status "Model files:"
        ls -lh models/ 2>/dev/null || echo "No models directory"
    fi
    
    # Check log file sizes
    if [ -d "logs" ]; then
        print_status "Log files:"
        ls -lh logs/ 2>/dev/null || echo "No logs directory"
    fi
}

# Function to show training log
show_training_log() {
    print_status "Showing training log..."
    cd "$PROJECT_DIR"
    docker exec zerbinetto-bot tail -f /app/training.log 2>/dev/null || print_warning "Training log not found. Training may not be running."
}

# Function to check training status
check_training_status() {
    print_status "Checking training status..."
    cd "$PROJECT_DIR"
    
    # Check if training process is running (using grep on logs since ps is not available in slim container)
    recent_training_activity=$(docker exec zerbinetto-bot tail -50 /app/training.log 2>/dev/null | grep "ML Engine selected move" | wc -l)
    if [ "$recent_training_activity" -gt 10 ]; then
        print_success "Training process is running (${recent_training_activity} recent moves)"
    else
        print_warning "Training process is not running"
    fi
    
    # Show recent training log entries
    print_status "Recent training log entries:"
    docker exec zerbinetto-bot tail -10 /app/training.log 2>/dev/null || print_warning "No training log found"
}

# Function to check comprehensive system status
check_system_status() {
    print_status "=== ZERBINETTO SYSTEM STATUS ==="
    cd "$PROJECT_DIR"
    
    # Check container status
    print_status "1. Container Status:"
    if docker ps | grep -q "zerbinetto-bot"; then
        container_status=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep zerbinetto-bot)
        print_success "Container: $container_status"
    else
        print_error "Container is not running!"
        return 1
    fi
    
    # Check bot activity
    print_status "2. Bot Activity:"
    recent_bot_logs=$(docker logs zerbinetto-bot --tail 5 2>/dev/null | grep -E "(Game|move|turn)" | tail -3)
    if [ -n "$recent_bot_logs" ]; then
        print_success "Bot is active - Recent activity:"
        echo "$recent_bot_logs" | while read line; do
            echo "   $line"
        done
    else
        print_warning "No recent bot activity found"
    fi
    
    # Check training status
    print_status "3. Training Status:"
    recent_training_activity=$(docker exec zerbinetto-bot tail -50 /app/training.log 2>/dev/null | grep "ML Engine selected move" | wc -l)
    if [ "$recent_training_activity" -gt 10 ]; then
        print_success "Training process is running (${recent_training_activity} recent moves)"
    else
        print_warning "Training process is not running"
    fi
    
    # Check training activity
    recent_training_logs=$(docker exec zerbinetto-bot tail -5 /app/training.log 2>/dev/null | grep "ML Engine selected move" | tail -3)
    if [ -n "$recent_training_logs" ]; then
        print_success "Training is active - Recent moves:"
        echo "$recent_training_logs" | while read line; do
            echo "   $line"
        done
    else
        print_warning "No recent training activity found"
    fi
    
    # Check model file
    print_status "4. Model Status:"
    if [ -f "models/chess_model.pkl" ]; then
        model_size=$(ls -lh models/chess_model.pkl | awk '{print $5}')
        model_time=$(ls -lh models/chess_model.pkl | awk '{print $6, $7, $8}')
        print_success "Model file exists: $model_size (last modified: $model_time)"
    else
        print_error "Model file not found!"
    fi
    
    # Check permissions
    print_status "5. Permissions:"
    if [ -r "models/chess_model.pkl" ] && [ -w "models/chess_model.pkl" ]; then
        print_success "Model file is readable and writable"
    else
        print_error "Model file permission issues!"
    fi
    
    # Check active games
    print_status "6. Active Games:"
    active_games=$(docker logs zerbinetto-bot --tail 50 2>/dev/null | grep "Game.*state update" | tail -3)
    if [ -n "$active_games" ]; then
        print_success "Active games detected:"
        echo "$active_games" | while read line; do
            echo "   $line"
        done
    else
        print_warning "No active games found"
    fi
    
    print_status "=== STATUS CHECK COMPLETE ==="
}

# Function to show help
show_help() {
    echo "Zerbinetto Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy      Full deployment (pull, cleanup, build, start, train)"
    echo "  update      Pull updates and restart with training (triggered by webhook)"
    echo "  restart     Restart the application"
    echo "  train       Start continuous training"
    echo "  stop        Stop the application"
    echo "  status      Show application status"
    echo "  logs        Show application logs"
    echo "  deploy-log  Show deployment log"
    echo "  training-log Show training log"
    echo "  training-status Check training status"
    echo "  status-full  Comprehensive system status check"
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
        print_status "Quick status check:"
        check_system_status
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
        start_training
        print_status "Quick status check:"
        check_system_status
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
        "status-full")
            check_system_status
            ;;
        "cleanup")
            cleanup_container
            ;;
        "disk-usage")
            check_disk_usage
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
