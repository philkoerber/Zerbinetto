#!/bin/bash

# Lichess Bot Account Upgrade Script (Shell Version)
# This script upgrades a Lichess account to bot status using the API endpoint

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Lichess Bot Account Upgrade Tool${NC}"
echo "========================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please create a .env file with your Lichess API token:"
    echo "  LICHESS_TOKEN=your_token_here"
    exit 1
fi

# Load token from .env file
source .env

if [ -z "$LICHESS_TOKEN" ]; then
    echo -e "${RED}❌ Error: LICHESS_TOKEN not found in .env file${NC}"
    echo "Please add your API token to the .env file:"
    echo "  LICHESS_TOKEN=your_token_here"
    exit 1
fi

echo -e "${YELLOW}📋 Instructions:${NC}"
echo "1. Create a new Lichess account (don't play any games)"
echo "2. Go to lichess.org/api → BOT → 'Upgrade BOT account'"
echo "3. Create a personal API token with all permissions"
echo "4. Add the token to your .env file"
echo "5. Run this script"
echo ""

echo -e "${YELLOW}🔍 Current token: ${LICHESS_TOKEN:0:10}...${NC}"
echo ""

# Ask for confirmation
read -p "🤔 Ready to upgrade your account to bot status? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🤖 Upgrading account to bot status...${NC}"
    
    # Make the API request
    response=$(curl -s -w "%{http_code}" -d '' \
        "https://lichess.org/api/bot/account/upgrade" \
        -H "Authorization: Bearer $LICHESS_TOKEN" \
        -H "Content-Type: application/json")
    
    # Extract status code (last 3 characters)
    status_code="${response: -3}"
    # Extract response body (everything except last 3 characters)
    response_body="${response%???}"
    
    if [ "$status_code" = "200" ]; then
        echo -e "${GREEN}✅ Success! Your account has been upgraded to bot status!${NC}"
        echo -e "${GREEN}🎉 You should now see 'BOT' in front of your username on your profile.${NC}"
        echo -e "${BLUE}🔗 Check your profile at: https://lichess.org/@/your_username${NC}"
    else
        echo -e "${RED}❌ Error: HTTP $status_code${NC}"
        echo -e "${RED}Response: $response_body${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}❌ Upgrade cancelled.${NC}"
fi
