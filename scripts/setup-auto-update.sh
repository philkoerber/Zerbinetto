#!/bin/bash

# Setup Auto-Update Script for Zerbinetto
# This script sets up automatic deployment from GitHub

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

# Configuration
PROJECT_DIR="/opt/zerbinetto"
DEPLOY_SCRIPT="$PROJECT_DIR/scripts/deploy.sh"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as a regular user."
   exit 1
fi

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory $PROJECT_DIR does not exist. Please run deployment first."
    exit 1
fi

# Check if deploy script exists
if [ ! -f "$DEPLOY_SCRIPT" ]; then
    print_error "Deploy script not found. Please ensure the project is properly cloned."
    exit 1
fi

# Make deploy script executable
print_status "Making deploy script executable..."
chmod +x "$DEPLOY_SCRIPT"

# Setup systemd service
print_status "Setting up systemd service..."
sudo cp "$PROJECT_DIR/config/zerbinetto-deploy.service" /etc/systemd/system/
sudo systemctl daemon-reload

# Setup cron job for periodic updates (every 30 minutes)
print_status "Setting up cron job for automatic updates..."
CRON_JOB="*/30 * * * * cd $PROJECT_DIR && $DEPLOY_SCRIPT update >> $PROJECT_DIR/cron.log 2>&1"

# Remove existing cron job if it exists
(crontab -l 2>/dev/null | grep -v "$PROJECT_DIR") | crontab -

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

print_success "Cron job added successfully"

# Create log files
touch "$PROJECT_DIR/cron.log"
touch "$PROJECT_DIR/deploy.log"

# Set proper permissions
chmod 644 "$PROJECT_DIR/cron.log"
chmod 644 "$PROJECT_DIR/deploy.log"

print_success "Auto-update setup completed!"
print_status ""
print_status "Setup includes:"
print_status "- Cron job that checks for updates every 30 minutes"
print_status "- Systemd service for manual deployment"
print_status "- Log files for monitoring"
print_status ""
print_status "To check status:"
print_status "  crontab -l                    # View cron jobs"
print_status "  tail -f $PROJECT_DIR/cron.log # View cron logs"
print_status "  tail -f $PROJECT_DIR/deploy.log # View deployment logs"
print_status ""
print_status "To manually trigger update:"
print_status "  $DEPLOY_SCRIPT update"
