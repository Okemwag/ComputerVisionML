#!/usr/bin/env python3
"""
Example Training Script: DeepLabV3+ for Classification

This script demonstrates training a DeepLabV3+ model for contrast detection classification
on medical CT scan images.

Requirements Addressed:
- 5.1: Visualize model predictions and performance metrics
- 5.2: Generate loss curves and metric plots
- 5.3: Create confusion matrices and ROC curves
- 5.4: Generate overlay visualizations with color-coded regions
"""

import os
import sys

sys.path.append("../")

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dataset import AugmentedMedicalDataset
from src.model import create_model

# Import custom modules
from src.train import create_trainer
from src.utils import save_config, setup_logging
from src.visualization import VisualizationEngine


def create_deeplabv3_classification_config():
    """Create configuration for DeepLabV3+ classification training."""
    config = {
        "experiment_name": "deeplabv3_classification_contrast_detection",
        # Data Configuration
        "data": {
            "data_dir": "../archive",
            "metadata_file": "../archive/overview.csv",
            "image_size": [256, 256],
            "batch_size": 12,  # Smaller batch size due to higher memory usage
            "num_workers": 4,
            "normalize_method": "hounsfield",
            "window_type": "soft_tissue",
        },
        # Model Configuration
        "model": {
            "architecture": "deeplabv3",
            "task_type": "classification",
            "in_channels": 1,
            "num_classes": 2,  # Contrast vs No-contrast
            "backbone": "custom",
            "dropout": 0.3,
        },
        # Training Configuration
        "training": {
            "num_epochs": 40,  # Fewer epochs due to more complex model
            "learning_rate": 0.0005,  # Lower learning rate
            "weight_decay": 1e-3,  # Higher weight decay
            "optimizer": "adamw",
            "scheduler": "cosine",
            "early_stopping_patience": 8,
            "gradient_accumulation_steps": 2,  # Accumulate gradients
            "mixed_precision": True,
        },
        # Augmentation Configuration
        "augmentation": {
            "rotation_limit": 10,  # Less aggressive augmentation
            "brightness_limit": 0.15,
            "contrast_limit": 0.15,
            "horizontal_flip": True,
            "vertical_flip": False,
            "gaussian_noise": 0.005,
            "augmentation_probability": 0.7,
        },
        # Loss Configuration
        "loss": {
            "type": "cross_entropy",
            "use_class_weights": True,
            "label_smoothing": 0.05,
        },
        # Paths Configuration
        "paths": {
            "output_dir": "../outputs/deeplabv3_classification",
            "checkpoint_dir": "../outputs/deeplabv3_classification/checkpoints",
            "logs_dir": "../outputs/deeplabv3_classification/logs",
            "visualizations_dir": "../outputs/deeplabv3_classification/visualizations",
        },
        # Device Configuration
        "device": {"use_cuda": True, "device_id": 0},
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ for Classification")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../outputs/deeplabv3_classification",
        help="Output directory",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone for transfer learning",
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = create_deeplabv3_classification_config()

    # Override config with command line arguments
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
        config["paths"]["checkpoint_dir"] = f"{args.output_dir}/checkpoints"
        config["paths"]["logs_dir"] = f"{args.output_dir}/logs"
        config["paths"]["visualizations_dir"] = f"{args.output_dir}/visualizations"

    # Create output directories
    for path_key, path_value in config["paths"].items():
        Path(path_value).mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(config["paths"]["logs_dir"])
    logger = logging.getLogger(__name__)

    logger.info("Starting DeepLabV3+ Classification Training")
    logger.info(f"Configuration: {config['experiment_name']}")

    # Save configuration
    config_path = Path(config["paths"]["output_dir"]) / "config.yaml"
    save_config(config, str(config_path))

    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset, val_dataset, test_dataset = (
            AugmentedMedicalDataset.create_augmented_datasets(
                data_dir=config["data"]["data_dir"],
                metadata_file=config["data"]["metadata_file"],
                target_size=tuple(config["data"]["image_size"]),
                normalize_method=config["data"]["normalize_method"],
                augmentation_config=config["augmentation"],
            )
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

        # Create model
        logger.info("Creating model...")
        model = create_model(config)

        # Freeze backbone if requested
        if args.freeze_backbone and hasattr(model, "freeze_backbone"):
            model.freeze_backbone()
            logger.info("Backbone frozen for transfer learning")

        logger.info(
            f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(config, device)

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Start training
        logger.info("Starting training...")
        training_history = trainer.fit(config["training"]["num_epochs"])

        # Create visualizations
        logger.info("Creating training visualizations...")
        viz_engine = VisualizationEngine(config["paths"]["visualizations_dir"])

        # Plot training curves
        viz_engine.plot_training_curves(training_history)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

        # Load best model for evaluation
        best_model_path = Path(config["paths"]["checkpoint_dir"]) / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
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

        # Print model comparison info
        logger.info("\nDeepLabV3+ Model Characteristics:")
        logger.info("- Uses atrous convolutions for multi-scale context")
        logger.info("- ASPP module captures features at multiple scales")
        logger.info("- More complex than U-Net but potentially more accurate")
        logger.info("- Higher memory usage and longer training time")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
