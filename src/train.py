"""
Medical Image Analysis - Training Pipeline

Main training orchestrator with support for classification and segmentation tasks.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .dataset import AugmentedMedicalDataset
from .losses import LossManager, create_loss_function
from .metrics import (
    CheckpointManager,
    MedicalMetrics,
    MetricsTracker,
    create_checkpoint_manager,
    create_metrics_tracker,
)
from .model import create_model


class Trainer:
    """Main training orchestrator for medical image analysis."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.logger = logging.getLogger(self.__class__.__name__)

        # Training configuration
        training_config = config.get("training", {})
        self.num_epochs = training_config.get("num_epochs", 50)
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        self.gradient_accumulation_steps = training_config.get(
            "gradient_accumulation_steps", 1
        )
        self.mixed_precision = training_config.get("mixed_precision", True)
        self.early_stopping_patience = training_config.get(
            "early_stopping_patience", 10
        )

        # Initialize optimizer
        self._setup_optimizer()

        # Initialize loss function
        self._setup_loss_function()

        # Initialize scheduler
        self._setup_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None

        # Initialize monitoring components
        self.metrics_tracker = create_metrics_tracker(config)
        self.checkpoint_manager = create_checkpoint_manager(config)
        self.medical_metrics = MedicalMetrics(
            num_classes=config.get("model", {}).get("num_classes", 2),
            task_type=config.get("model", {}).get("task_type", "classification"),
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.logger.info(
            f"Trainer initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        training_config = self.config.get("training", {})
        optimizer_name = training_config.get("optimizer", "adam").lower()

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.logger.info(f"Using {optimizer_name.upper()} optimizer")

    def _setup_loss_function(self):
        """Setup loss function based on task type."""
        model_config = self.config.get("model", {})
        task_type = model_config.get("task_type", "classification")

        # Get class weights if available
        class_weights = None
        if hasattr(self.train_loader.dataset, "get_class_weights"):
            class_weights = self.train_loader.dataset.get_class_weights().to(
                self.device
            )

        # Create loss manager
        self.loss_manager = create_loss_function(task_type, self.config, class_weights)

        # For backward compatibility, set criterion
        self.criterion = (
            self.loss_manager.criterion
            if hasattr(self.loss_manager, "criterion")
            else self.loss_manager
        )

        self.logger.info(f"Loss function setup for {task_type} task")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        training_config = self.config.get("training", {})
        scheduler_name = training_config.get("scheduler", "reduce_on_plateau").lower()

        if scheduler_name == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
        elif scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.1
            )
        else:
            self.scheduler = None

        if self.scheduler:
            self.logger.info(f"Using {scheduler_name} scheduler")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels, ages, sample_ids) in enumerate(
            self.train_loader
        ):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Log progress
            if batch_idx % 10 == 0:
                self.logger.debug(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct_predictions / total_samples

        return {"loss": avg_loss, "accuracy": accuracy}

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels, ages, sample_ids in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct_predictions / total_samples

        return {"loss": avg_loss, "accuracy": accuracy}

    def fit(self, num_epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train (uses config if None)

        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.num_epochs

        self.logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate_epoch()

            # Update metrics tracking
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics_tracker.update("train", train_metrics, epoch, current_lr)
            self.metrics_tracker.update("val", val_metrics, epoch)
            self.metrics_tracker.add_epoch_time(time.time() - epoch_start_time)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Update best metrics and check for improvement
            self.checkpoint_manager.update_best_metrics(val_metrics, epoch)
            is_best = self.checkpoint_manager.is_best_model(val_metrics, "loss")

            # Early stopping check
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch,
                val_metrics,
                self.config,
                is_best,
            )

            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}% - "
                f"Time: {epoch_time:.2f}s"
            )

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")

        # Save final metrics
        self.metrics_tracker.save_metrics()

        return self.metrics_tracker.metrics_history

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint_dir = Path(
            self.config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "metrics": metrics,
            "config": self.config,
            "training_history": self.training_history,
        }

        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"New best model saved with validation loss: {metrics['loss']:.4f}"
            )

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.training_history = checkpoint.get(
            "training_history", self.training_history
        )

        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")

        return checkpoint


def create_trainer(config: Dict[str, Any], device: str = "cuda") -> Trainer:
    """
    Factory function to create trainer from configuration.

    Args:
        config: Configuration dictionary
        device: Device to use for training

    Returns:
        Trainer instance
    """
    # Create model
    model = create_model(config)

    # Create datasets
    data_config = config.get("data", {})
    train_dataset, val_dataset, _ = AugmentedMedicalDataset.create_augmented_datasets(
        data_dir=data_config.get("data_dir", "data"),
        metadata_file=data_config.get("metadata_file", "data/metadata.csv"),
        target_size=tuple(config.get("model", {}).get("image_size", [256, 256])),
        normalize_method=config.get("model", {}).get("normalize_method", "hounsfield"),
        augmentation_config=config.get("augmentation", {}),
    )

    # Create data loaders
    batch_size = data_config.get("batch_size", 16)
    num_workers = data_config.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)

    return trainer
