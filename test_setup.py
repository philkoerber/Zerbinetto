#!/usr/bin/env python3
"""
Test script to verify bot setup and dependencies.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported.
    
    Args:
        module_name: Name of the module to import
        package_name: Display name for the package (defaults to module_name)
    
    Returns:
        True if import successful, False otherwise
    """
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Lichess Bot Setup Test ===\n")
    
    # Test Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    else:
        print("✓ Python version is compatible")
    
    print("\nTesting dependencies:")
    
    # Test required packages
    dependencies = [
        ("requests", "requests"),
        ("dotenv", "python-dotenv"),
        ("websockets", "websockets"),
    ]
    
    all_good = True
    for module, package in dependencies:
        if not test_import(module, package):
            all_good = False
    
    print("\nTesting bot modules:")
    
    # Test bot modules
    bot_modules = [
        ("lichess_client", "lichess_client"),
        ("game_handler", "game_handler"),
    ]
    
    for module, package in bot_modules:
        if not test_import(module, package):
            all_good = False
    
    print("\nTesting environment:")
    
    # Test environment variables
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    token = os.getenv('LICHESS_TOKEN')
    if token:
        print("✓ LICHESS_TOKEN environment variable is set")
    else:
        print("✗ LICHESS_TOKEN environment variable is not set")
        print("  Create a .env file with your Lichess API token")
        all_good = False
    
    print("\n=== Test Results ===")
    
    if all_good:
        print("✓ All tests passed! The bot should be ready to run.")
        print("\nTo start the bot:")
        print("  python3 bot.py")
        return True
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
