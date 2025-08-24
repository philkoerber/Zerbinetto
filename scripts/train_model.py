#!/usr/bin/env python3
"""
Standalone Model Training Script

This script can be run independently to train the chess model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import time
from src.trainer import SelfPlayTrainer

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train chess model through self-play')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--games-per-iteration', type=int, default=50, help='Games per iteration')
    parser.add_argument('--model-path', default='models/chess_model.pkl', help='Model file path')
    parser.add_argument('--continuous', action='store_true', help='Run continuous training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting chess model training")
    
    # Create trainer
    trainer = SelfPlayTrainer(args.model_path)
    trainer.games_per_iteration = args.games_per_iteration
    
    try:
        if args.continuous:
            # Run continuous training
            logger.info(f"Running continuous training for {args.iterations} iterations")
            trainer.continuous_training(args.iterations)
        else:
            # Run single training iteration
            logger.info("Running single training iteration")
            trainer.train_iteration()
        
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
