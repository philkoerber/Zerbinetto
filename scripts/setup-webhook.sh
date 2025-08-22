#!/bin/bash

# Setup Webhook Script for Zerbinetto
# This script sets up the GitHub webhook server for automatic deployment

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
WEBHOOK_SCRIPT="$PROJECT_DIR/scripts/webhook_server.py"
WEBHOOK_SERVICE="$PROJECT_DIR/config/zerbinetto-webhook.service"
WEBHOOK_PORT=9001

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

# Check if webhook script exists
if [ ! -f "$WEBHOOK_SCRIPT" ]; then
    print_error "Webhook script not found. Please ensure the project is properly cloned."
    exit 1
fi

# Make webhook script executable
print_status "Making webhook script executable..."
chmod +x "$WEBHOOK_SCRIPT"

# Setup systemd service
print_status "Setting up systemd service for webhook server..."
sudo cp "$WEBHOOK_SERVICE" /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start the webhook service
print_status "Starting webhook service..."
sudo systemctl enable zerbinetto-webhook
sudo systemctl start zerbinetto-webhook

# Check if service is running
sleep 2
if sudo systemctl is-active --quiet zerbinetto-webhook; then
    print_success "Webhook service is running"
else
    print_error "Failed to start webhook service"
    sudo systemctl status zerbinetto-webhook
    exit 1
fi

# Check if port is listening
if netstat -tuln | grep -q ":$WEBHOOK_PORT "; then
    print_success "Webhook server is listening on port $WEBHOOK_PORT"
else
    print_warning "Webhook server may not be listening on port $WEBHOOK_PORT"
fi

# Get VM IP address for GitHub webhook configuration
VM_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VM_IP_ADDRESS")

print_success "Webhook setup completed!"
print_status ""
print_status "Next steps:"
print_status "1. Go to your GitHub repository: https://github.com/philkoerber/Zerbinetto"
print_status "2. Go to Settings → Webhooks → Add webhook"
print_status "3. Configure the webhook:"
print_status "   - Payload URL: http://$VM_IP:$WEBHOOK_PORT"
print_status "   - Content type: application/json"
print_status "   - Secret: zerbinetto_webhook_secret"
print_status "   - Events: Just the push event"
print_status "   - Active: ✓"
print_status ""
print_status "To check webhook status:"
print_status "  sudo systemctl status zerbinetto-webhook"
print_status "  sudo journalctl -u zerbinetto-webhook -f"
print_status "  tail -f $PROJECT_DIR/webhook.log"
print_status ""
print_status "To test the webhook:"
print_status "  curl http://localhost:$WEBHOOK_PORT"
print_status ""
print_status "Webhook server is now ready to receive GitHub push events!"
