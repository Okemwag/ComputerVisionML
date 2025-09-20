#!/usr/bin/env python3
"""
Example Training Script: U-Net for Classification

This script demonstrates training a U-Net model for contrast detection classification
on medical CT scan images.

Requirements Addressed:
- 5.1: Visualize model predictions and performance metrics
- 5.2: Generate loss curves and metric plots
- 5.3: Create confusion matrices and ROC curves
- 5.4: Generate overlay visualizations with color-coded regions
"""

import sys
import os
sys.path.append('../')

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import logging

# Import custom modules
from src.train import create_trainer
from src.model import create_model
from src.dataset import AugmentedMedicalDataset
from src.utils import setup_logging, save_config
from src.visualization import VisualizationEngine

def create_unet_classification_config():
    """Create configuration for U-Net classification training."""
    config = {
        'experiment_name': 'unet_classification_contrast_detection',
        
        # Data Configuration
        'data': {
            'data_dir': '../archive',
            'metadata_file': '../archive/overview.csv',
            'image_size': [256, 256],
            'batch_size': 16,
            'num_workers': 4,
            'normalize_method': 'hounsfield',
            'window_type': 'soft_tissue'
        },
        
        # Model Configuration
        'model': {
            'architecture': 'unet',
            'task_type': 'classification',
            'in_channels': 1,
            'num_classes': 2,  # Contrast vs No-contrast
            'depth': 4,
            'start_filters': 64,
            'bilinear': True,
            'dropout': 0.2
        },
        
        # Training Configuration
        'training': {
            'num_epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
            'early_stopping_patience': 10,
            'gradient_accumulation_steps': 1,
            'mixed_precision': True
        },
        
        # Augmentation Configuration
        'augmentation': {
            'rotation_limit': 15,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'horizontal_flip': True,
            'vertical_flip': False,
            'gaussian_noise': 0.01,
            'augmentation_probability': 0.8
        },
        
        # Loss Configuration
        'loss': {
            'type': 'cross_entropy',
            'use_class_weights': True,
            'label_smoothing': 0.1
        },
        
        # Paths Configuration
        'paths': {
            'output_dir': '../outputs/unet_classification',
            'checkpoint_dir': '../outputs/unet_classification/checkpoints',
            'logs_dir': '../outputs/unet_classification/logs',
            'visualizations_dir': '../outputs/unet_classification/visualizations'
        },
        
        # Device Configuration
        'device': {
            'use_cuda': True,
            'device_id': 0
        }
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Train U-Net for Classification')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output-dir', type=str, default='../outputs/unet_classification', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_unet_classification_config()
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
        config['paths']['checkpoint_dir'] = f"{args.output_dir}/checkpoints"
        config['paths']['logs_dir'] = f"{args.output_dir}/logs"
        config['paths']['visualizations_dir'] = f"{args.output_dir}/visualizations"
    
    # Create output directories
    for path_key, path_value in config['paths'].items():
        Path(path_value).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(config['paths']['logs_dir'])
    logger = logging.getLogger(__name__)
    
    logger.info("Starting U-Net Classification Training")
    logger.info(f"Configuration: {config['experiment_name']}")
    
    # Save configuration
    config_path = Path(config['paths']['output_dir']) / 'config.yaml'
    save_config(config, str(config_path))
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    try:
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset, val_dataset, test_dataset = AugmentedMedicalDataset.create_augmented_datasets(
            data_dir=config['data']['data_dir'],
            metadata_file=config['data']['metadata_file'],
            target_size=tuple(config['data']['image_size']),
            normalize_method=config['data']['normalize_method'],
            augmentation_config=config['augmentation']
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(config, device)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        logger.info("Starting training...")
        training_history = trainer.fit(config['training']['num_epochs'])
        
        # Create visualizations
        logger.info("Creating training visualizations...")
        viz_engine = VisualizationEngine(config['paths']['visualizations_dir'])
        
        # Plot training curves
        viz_engine.plot_training_curves(training_history)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        # Load best model for evaluation
        best_model_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model for evaluation")
        
        # Generate predictions and visualizations
        model.eval()
        all_predictions = []
        all_targets = []
        all_images = []
        
        with torch.no_grad():
            for i, (images, targets, ages, sample_ids) in enumerate(test_loader):
                if i >= 5:  # Limit to first few batches for visualization
                    break
                    
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                predictions = torch.softmax(outputs, dim=1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_images.append(images.cpu())
        
        if all_predictions:
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            all_images = torch.cat(all_images)
            
            # Create prediction visualizations
            viz_engine.visualize_predictions(
                all_images[:16], all_predictions[:16], all_targets[:16], num_samples=16
            )
            
            # Create confusion matrix and ROC curve
            viz_engine.plot_classification_metrics(all_predictions, all_targets)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {config['paths']['output_dir']}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()