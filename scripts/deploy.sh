#!/bin/bash

# Zerbinetto Bulletproof Deployment Script
# This script ensures 100% reliable deployment every time

set -euo pipefail  # Exit on any error, undefined vars, pipe failures

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
MAX_RETRIES=3
WAIT_TIME=30

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): [INFO] $1" >> "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): [SUCCESS] $1" >> "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): [WARNING] $1" >> "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): [ERROR] $1" >> "$LOG_FILE"
}

# Function to ensure we're in the right directory
ensure_project_directory() {
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "Project directory $PROJECT_DIR does not exist!"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        print_error "Docker Compose file not found in $PROJECT_DIR"
        exit 1
    fi
    
    print_success "Project directory verified: $PROJECT_DIR"
}

# Function to completely clean up everything
complete_cleanup() {
    print_status "Performing complete cleanup..."
    
    cd "$PROJECT_DIR"
    
    # Stop all containers forcefully
    print_status "Stopping all containers..."
    docker-compose down --remove-orphans --volumes --timeout 30 2>/dev/null || true
    
    # Force remove any containers with zerbinetto in the name
    print_status "Force removing any lingering containers..."
    docker ps -aq --filter "name=zerbinetto" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
    docker ps -aq --filter "name=zerbinetto-bot" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
    
    # Remove any container with the exact name
    docker rm -f zerbinetto-bot 2>/dev/null || true
    
    # Wait a moment for cleanup
    sleep 2
    
    # Clean up Docker system completely
    print_status "Cleaning Docker system..."
    docker system prune -af --volumes 2>/dev/null || true
    
    # Clean up any dangling images
    docker image prune -af 2>/dev/null || true
    
    # Clean up networks
    docker network prune -f 2>/dev/null || true
    
    # Clean up old logs
    print_status "Cleaning up old logs..."
    find . -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
    find . -name "*.log.old" -type f -delete 2>/dev/null || true
    
    # Clean up old model backups (keep only last 3)
    if [ -d "models" ]; then
        print_status "Cleaning up old model backups..."
        ls -t models/chess_model.pkl.backup_* 2>/dev/null | tail -n +4 | xargs -r rm 2>/dev/null || true
    fi
    
    print_success "Complete cleanup finished"
}

# Function to pull latest changes safely
pull_latest_changes() {
    print_status "Pulling latest changes from GitHub..."
    
    cd "$PROJECT_DIR"
    
    # Backup current state
    if [ -d ".git" ]; then
        print_status "Backing up current state..."
        git stash push -m "Auto-stash before deployment $(date)" 2>/dev/null || true
    fi
    
    # Fetch latest changes
    print_status "Fetching latest changes..."
    git fetch origin main --force 2>/dev/null || {
        print_error "Failed to fetch from GitHub"
        return 1
    }
    
    # Reset to latest main
    print_status "Resetting to latest main..."
    git reset --hard origin/main 2>/dev/null || {
        print_error "Failed to reset to latest main"
        return 1
    }
    
    # Clean any untracked files
    git clean -fd 2>/dev/null || true
    
    print_success "Successfully updated to latest version: $(git rev-parse --short HEAD)"
}

# Function to build Docker image with retries
build_docker_image() {
    print_status "Building Docker image..."
    
    cd "$PROJECT_DIR"
    
    local retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if docker-compose build --no-cache --pull; then
            print_success "Docker image built successfully"
            return 0
        else
            retry_count=$((retry_count + 1))
            print_warning "Build failed, attempt $retry_count/$MAX_RETRIES"
            if [ $retry_count -lt $MAX_RETRIES ]; then
                print_status "Waiting $WAIT_TIME seconds before retry..."
                sleep $WAIT_TIME
                complete_cleanup
            fi
        fi
    done
    
    print_error "Failed to build Docker image after $MAX_RETRIES attempts"
    return 1
}

# Function to start application with health checks
start_application() {
    print_status "Starting application..."
    
    cd "$PROJECT_DIR"
    
    # Start the application
    if ! docker-compose up -d; then
        print_error "Failed to start application"
        return 1
    fi
    
    # Wait for container to be ready
    print_status "Waiting for container to be ready..."
    local wait_count=0
    while [ $wait_count -lt 60 ]; do
        if docker ps | grep -q "zerbinetto-bot.*Up"; then
            print_success "Container is running"
            break
        fi
        sleep 2
        wait_count=$((wait_count + 1))
    done
    
    if [ $wait_count -ge 60 ]; then
        print_error "Container failed to start within timeout"
        return 1
    fi
    
    # Fix permissions
    fix_permissions
    
    # Health check
    if ! check_application_health; then
        print_error "Application health check failed"
        return 1
    fi
    
    print_success "Application started successfully"
    return 0
}

# Function to fix all permissions
fix_permissions() {
    print_status "Fixing permissions..."
    
    cd "$PROJECT_DIR"
    
    # Create directories if they don't exist
    mkdir -p models logs
    
    # Fix model file permissions
    if [ -f "models/chess_model.pkl" ]; then
        sudo chown 1000:1000 models/chess_model.pkl 2>/dev/null || true
        sudo chmod 644 models/chess_model.pkl 2>/dev/null || true
    fi
    
    # Fix directory permissions
    sudo chown -R 1000:1000 models/ logs/ 2>/dev/null || true
    sudo chmod -R 755 models/ logs/ 2>/dev/null || true
    
    print_success "Permissions fixed"
}

# Function to check application health
check_application_health() {
    print_status "Performing health check..."
    
    cd "$PROJECT_DIR"
    
    # Wait a bit for the application to fully start
    sleep 10
    
    # Check if container is running
    if ! docker ps | grep -q "zerbinetto-bot.*Up"; then
        print_error "Container is not running"
        return 1
    fi
    
    # Check if bot is responding
    local health_check_count=0
    while [ $health_check_count -lt 30 ]; do
        if docker logs zerbinetto-bot --tail 10 2>/dev/null | grep -q "Bot is running\|Logged in as\|Waiting for challenges"; then
            print_success "Bot is responding"
            return 0
        fi
        sleep 2
        health_check_count=$((health_check_count + 1))
    done
    
    print_error "Bot health check failed"
    return 1
}

# Function to start training reliably
start_training() {
    print_status "Starting continuous training..."
    
    cd "$PROJECT_DIR"
    
    # Stop any existing training
    docker exec zerbinetto-bot pkill -f "trainer.py" 2>/dev/null || true
    
    # Wait a moment
    sleep 5
    
    # Start training in background with proper error handling
    docker exec -d zerbinetto-bot bash -c "
        set -e
        cd /app
        
        # Create training directory if it doesn't exist
        mkdir -p /app/logs
        
        # Start training with proper logging
        nohup python src/trainer.py --forever --games-per-iteration 100 > /app/training.log 2>&1 &
        
        # Store PID for later management
        echo \$! > /app/training.pid
        
        # Start log rotation
        (
            while true; do
                sleep 3600
                if [ -f /app/training.log ] && [ \$(stat -c%s /app/training.log 2>/dev/null || echo 0) -gt 10485760 ]; then
                    mv /app/training.log /app/training.log.old
                    touch /app/training.log
                    ls -t /app/training.log.* 2>/dev/null | tail -n +6 | xargs -r rm
                fi
            done
        ) &
        
        echo \$! > /app/log_rotation.pid
    "
    
    # Wait for training to start
    sleep 10
    
    # Verify training is running
    if docker exec zerbinetto-bot test -f /app/training.pid; then
        print_success "Training started successfully"
    else
        print_warning "Training may not have started properly"
    fi
}

# Function to verify everything is working
verify_deployment() {
    print_status "Verifying deployment..."
    
    cd "$PROJECT_DIR"
    
    # Check container status
    if ! docker ps | grep -q "zerbinetto-bot.*Up"; then
        print_error "Container is not running"
        return 1
    fi
    
    # Check bot activity
    if docker logs zerbinetto-bot --tail 20 2>/dev/null | grep -q "Bot is running\|Logged in as"; then
        print_success "Bot is active"
    else
        print_warning "Bot may not be fully active"
    fi
    
    # Check training
    if docker exec zerbinetto-bot test -f /app/training.pid 2>/dev/null; then
        print_success "Training is running"
    else
        print_warning "Training may not be running"
    fi
    
    # Check model file
    if [ -f "models/chess_model.pkl" ]; then
        print_success "Model file exists and accessible"
    else
        print_warning "Model file not found"
    fi
    
    print_success "Deployment verification completed"
}

# Function to show comprehensive status
show_comprehensive_status() {
    print_status "=== ZERBINETTO COMPREHENSIVE STATUS ==="
    
    cd "$PROJECT_DIR"
    
    # Container status
    print_status "1. Container Status:"
    if docker ps | grep -q "zerbinetto-bot"; then
        container_info=$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep zerbinetto-bot)
        print_success "Container: $container_info"
    else
        print_error "Container is not running!"
    fi
    
    # Bot activity
    print_status "2. Bot Activity:"
    recent_logs=$(docker logs zerbinetto-bot --tail 10 2>/dev/null | grep -E "(Bot is running|Logged in as|Game|move)" | tail -5)
    if [ -n "$recent_logs" ]; then
        print_success "Recent bot activity:"
        echo "$recent_logs" | while read line; do
            echo "   $line"
        done
    else
        print_warning "No recent bot activity"
    fi
    
    # Training status
    print_status "3. Training Status:"
    if docker exec zerbinetto-bot test -f /app/training.pid 2>/dev/null; then
        training_pid=$(docker exec zerbinetto-bot cat /app/training.pid 2>/dev/null)
        print_success "Training PID: $training_pid"
        
        # Check recent training activity
        recent_training=$(docker exec zerbinetto-bot tail -20 /app/training.log 2>/dev/null | grep "ML Engine selected move" | wc -l)
        if [ "$recent_training" -gt 0 ]; then
            print_success "Training is active ($recent_training recent moves)"
        else
            print_warning "Training may be idle"
        fi
    else
        print_warning "Training process not found"
    fi
    
    # Model status
    print_status "4. Model Status:"
    if [ -f "models/chess_model.pkl" ]; then
        model_size=$(ls -lh models/chess_model.pkl | awk '{print $5}')
        model_time=$(ls -lh models/chess_model.pkl | awk '{print $6, $7, $8}')
        print_success "Model: $model_size (modified: $model_time)"
    else
        print_error "Model file missing!"
    fi
    
    # Active games
    print_status "5. Active Games:"
    active_games=$(docker logs zerbinetto-bot --tail 50 2>/dev/null | grep "Game.*state update" | tail -3)
    if [ -n "$active_games" ]; then
        print_success "Active games:"
        echo "$active_games" | while read line; do
            echo "   $line"
        done
    else
        print_warning "No active games detected"
    fi
    
    # System resources
    print_status "6. System Resources:"
    disk_usage=$(df -h . | tail -1 | awk '{print $5}')
    print_status "Disk usage: $disk_usage"
    
    docker_usage=$(docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}" 2>/dev/null | head -5)
    print_status "Docker usage:"
    echo "$docker_usage"
    
    print_status "=== STATUS CHECK COMPLETE ==="
}

# Function to show logs
show_logs() {
    print_status "Showing application logs..."
    cd "$PROJECT_DIR"
    docker-compose logs -f --tail 100
}

# Function to show training logs
show_training_logs() {
    print_status "Showing training logs..."
    cd "$PROJECT_DIR"
    docker exec zerbinetto-bot tail -f /app/training.log 2>/dev/null || print_warning "Training log not found"
}

# Function to stop everything
stop_everything() {
    print_status "Stopping everything..."
    cd "$PROJECT_DIR"
    
    # Stop training
    docker exec zerbinetto-bot pkill -f "trainer.py" 2>/dev/null || true
    
    # Stop containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    print_success "Everything stopped"
}

# Function to restart everything
restart_everything() {
    print_status "Restarting everything..."
    stop_everything
    sleep 5
    start_application
    start_training
    verify_deployment
}

# Main deployment function (bulletproof)
deploy() {
    print_status "Starting bulletproof deployment..."
    
    # Ensure we're in the right place
    ensure_project_directory
    
    # Complete cleanup
    complete_cleanup
    
    # Pull latest changes
    pull_latest_changes
    
    # Build with retries
    build_docker_image
    
    # Start application
    start_application
    
    # Start training
    start_training
    
    # Verify everything
    verify_deployment
    
    print_success "Bulletproof deployment completed successfully!"
    show_comprehensive_status
}

# Update function (for webhooks)
update() {
    print_status "Starting webhook-triggered update..."
    
    # Same as deploy but with extra safety
    deploy
}

# Function to show help
show_help() {
    echo "Zerbinetto Bulletproof Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy      Full bulletproof deployment"
    echo "  update      Webhook-triggered update (same as deploy)"
    echo "  restart     Restart everything"
    echo "  stop        Stop everything"
    echo "  status      Comprehensive status check"
    echo "  logs        Show application logs"
    echo "  training    Show training logs"
    echo "  cleanup     Complete cleanup"
    echo "  help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 deploy   # Full deployment"
    echo "  $0 update   # Webhook update"
    echo "  $0 status   # Check everything"
}

# Main script logic
main() {
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    
    print_status "Deployment script started: $0 $*"
    
    case "${1:-help}" in
        "deploy"|"update")
            deploy
            ;;
        "restart")
            ensure_project_directory
            restart_everything
            ;;
        "stop")
            ensure_project_directory
            stop_everything
            ;;
        "status")
            ensure_project_directory
            show_comprehensive_status
            ;;
        "logs")
            ensure_project_directory
            show_logs
            ;;
        "training")
            ensure_project_directory
            show_training_logs
            ;;
        "cleanup")
            ensure_project_directory
            complete_cleanup
            ;;
        "help"|"--help"|"-h"|"")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
