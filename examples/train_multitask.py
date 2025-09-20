#!/usr/bin/env python3
"""
Example Training Script: Multi-task Learning

This script demonstrates training models for both classification and segmentation tasks
simultaneously on medical CT scan images.

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

def create_multitask_config():
    """Create configuration for multi-task learning."""
    config = {
        'experiment_name': 'multitask_classification_segmentation',
        
        # Data Configuration
        'data': {
            'data_dir': '../archive',
            'metadata_file': '../archive/overview.csv',
            'image_size': [256, 256],
            'batch_size': 8,  # Smaller batch size for multi-task learning
            'num_workers': 4,
            'normalize_method': 'hounsfield',
            'window_type': 'soft_tissue'
        },
        
        # Model Configuration
        'model': {
            'architecture': 'unet',  # U-Net works well for multi-task
            'task_type': 'both',  # Both classification and segmentation
            'in_channels': 1,
            'num_classes': 2,  # For both tasks
            'depth': 4,
            'start_filters': 64,
            'bilinear': True,
            'dropout': 0.25
        },
        
        # Training Configuration
        'training': {
            'num_epochs': 60,  # More epochs for multi-task learning
            'learning_rate': 0.0008,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau',
            'early_stopping_patience': 12,
            'gradient_accumulation_steps': 2,
            'mixed_precision': True
        },
        
        # Augmentation Configuration
        'augmentation': {
            'rotation_limit': 12,
            'brightness_limit': 0.18,
            'contrast_limit': 0.18,
            'horizontal_flip': True,
            'vertical_flip': False,
            'gaussian_noise': 0.008,
            'augmentation_probability': 0.75
        },
        
        # Loss Configuration
        'loss': {
            'classification_loss': 'cross_entropy',
            'segmentation_loss': 'dice',
            'loss_weights': {
                'classification': 1.0,
                'segmentation': 2.0  # Higher weight for segmentation
            },
            'use_class_weights': True,
            'label_smoothing': 0.08
        },
        
        # Paths Configuration
        'paths': {
            'output_dir': '../outputs/multitask_learning',
            'checkpoint_dir': '../outputs/multitask_learning/checkpoints',
            'logs_dir': '../outputs/multitask_learning/logs',
            'visualizations_dir': '../outputs/multitask_learning/visualizations'
        },
        
        # Device Configuration
        'device': {
            'use_cuda': True,
            'device_id': 0
        }
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Train Multi-task Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0008, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output-dir', type=str, default='../outputs/multitask_learning', 
                       help='Output directory')
    parser.add_argument('--cls-weight', type=float, default=1.0, 
                       help='Weight for classification loss')
    parser.add_argument('--seg-weight', type=float, default=2.0, 
                       help='Weight for segmentation loss')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_multitask_config()
    
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
    
    # Update loss weights
    config['loss']['loss_weights']['classification'] = args.cls_weight
    config['loss']['loss_weights']['segmentation'] = args.seg_weight
    
    # Create output directories
    for path_key, path_value in config['paths'].items():
        Path(path_value).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(config['paths']['logs_dir'])
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Multi-task Learning Training")
    logger.info(f"Configuration: {config['experiment_name']}")
    logger.info(f"Classification weight: {args.cls_weight}")
    logger.info(f"Segmentation weight: {args.seg_weight}")
    
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
        logger.info("Creating multi-task model...")
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
        logger.info("Starting multi-task training...")
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
        all_cls_predictions = []
        all_seg_predictions = []
        all_targets = []
        all_images = []
        
        with torch.no_grad():
            for i, (images, targets, ages, sample_ids) in enumerate(test_loader):
                if i >= 5:  # Limit to first few batches for visualization
                    break
                    
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                
                if isinstance(outputs, dict):
                    # Multi-task output
                    cls_predictions = torch.softmax(outputs['classification'], dim=1)
                    seg_predictions = torch.softmax(outputs['segmentation'], dim=1)
                    
                    all_cls_predictions.append(cls_predictions.cpu())
                    all_seg_predictions.append(seg_predictions.cpu())
                else:
                    # Single task output (fallback)
                    predictions = torch.softmax(outputs, dim=1)
                    all_cls_predictions.append(predictions.cpu())
                
                all_targets.append(targets.cpu())
                all_images.append(images.cpu())
        
        if all_cls_predictions:
            all_cls_predictions = torch.cat(all_cls_predictions)
            all_targets = torch.cat(all_targets)
            all_images = torch.cat(all_images)
            
            # Create classification visualizations
            viz_engine.visualize_predictions(
                all_images[:16], all_cls_predictions[:16], all_targets[:16], num_samples=16
            )
            
            # Create confusion matrix and ROC curve for classification
            viz_engine.plot_classification_metrics(all_cls_predictions, all_targets)
            
            # Create segmentation visualizations if available
            if all_seg_predictions:
                all_seg_predictions = torch.cat(all_seg_predictions)
                
                # Create segmentation overlays
                for i in range(min(8, len(all_images))):
                    overlay = viz_engine.create_segmentation_overlay(
                        all_images[i], all_seg_predictions[i].argmax(dim=0)
                    )
                    viz_engine.save_image(overlay, f'segmentation_overlay_{i}.png')
        
        logger.info("Multi-task training completed successfully!")
        logger.info(f"Results saved to: {config['paths']['output_dir']}")
        
        # Print multi-task learning insights
        logger.info("\nMulti-task Learning Insights:")
        logger.info("- Shared representations can improve both tasks")
        logger.info("- Classification helps with global understanding")
        logger.info("- Segmentation provides detailed spatial information")
        logger.info("- Loss weighting is crucial for balanced learning")
        logger.info("- May require more training time to converge")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()