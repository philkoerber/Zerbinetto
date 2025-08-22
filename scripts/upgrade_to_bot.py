#!/usr/bin/env python3
"""
Lichess Bot Account Upgrade Script

This script upgrades a Lichess account to bot status using the API endpoint.
Follow these steps:
1. Create a new Lichess account (don't play any games on it)
2. Go to lichess.org/api → BOT → "Upgrade BOT account"
3. Create a personal API token with all permissions checked
4. Put the token in your .env file
5. Run this script
"""

import os
import requests
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upgrade_to_bot():
    """Upgrade account to bot status using Lichess API."""
    
    # Get token from environment
    token = os.getenv('LICHESS_TOKEN')
    if not token:
        print("❌ Error: LICHESS_TOKEN not found in environment variables")
        print("Please add your API token to the .env file:")
        print("LICHESS_TOKEN=your_token_here")
        return False
    
    # API endpoint for bot upgrade
    url = "https://lichess.org/api/bot/account/upgrade"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print("🤖 Upgrading account to bot status...")
    print(f"📡 Making request to: {url}")
    
    try:
        # Make the upgrade request
        response = requests.post(url, headers=headers, data='')
        
        if response.status_code == 200:
            print("✅ Success! Your account has been upgraded to bot status!")
            print("🎉 You should now see 'BOT' in front of your username on your profile.")
            print("🔗 Check your profile at: https://lichess.org/@/your_username")
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

def verify_bot_status():
    """Verify that the account has bot status."""
    
    token = os.getenv('LICHESS_TOKEN')
    if not token:
        print("❌ Error: LICHESS_TOKEN not found")
        return False
    
    url = "https://lichess.org/api/account"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            account_data = response.json()
            username = account_data.get('username', 'Unknown')
            is_bot = account_data.get('title', '') == 'BOT'
            
            print(f"👤 Account: {username}")
            if is_bot:
                print("✅ Bot status confirmed!")
            else:
                print("❌ Bot status not found. Try running the upgrade script first.")
            return is_bot
        else:
            print(f"❌ Error checking account: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    print("🤖 Lichess Bot Account Upgrade Tool")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        print("🔍 Verifying bot status...")
        verify_bot_status()
    else:
        print("📋 Instructions:")
        print("1. Create a new Lichess account (don't play any games)")
        print("2. Go to lichess.org/api → BOT → 'Upgrade BOT account'")
        print("3. Create a personal API token with all permissions")
        print("4. Add the token to your .env file")
        print("5. Run this script")
        print()
        
        # Check if token exists
        token = os.getenv('LICHESS_TOKEN')
        if not token:
            print("❌ No API token found in .env file")
            print("Please add your token to the .env file first.")
            return
        
        # Ask for confirmation
        response = input("🤔 Ready to upgrade your account to bot status? (y/N): ")
        if response.lower() in ['y', 'yes']:
            if upgrade_to_bot():
                print("\n🎉 Upgrade completed! You can now verify with:")
                print("python upgrade_to_bot.py verify")
        else:
            print("❌ Upgrade cancelled.")

if __name__ == "__main__":
    main()
