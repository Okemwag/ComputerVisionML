"""
Medical Image Analysis - Metrics and Monitoring

Comprehensive metrics calculation and training monitoring utilities.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, save_dir: str = "outputs/logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = {
            "train": {
                "loss": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
            },
            "val": {
                "loss": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
            },
        }

        self.epoch_times = []
        self.learning_rates = []

        self.logger = logging.getLogger(self.__class__.__name__)

    def update(
        self, phase: str, metrics: Dict[str, float], epoch: int, lr: float = None
    ):
        """Update metrics for current epoch."""
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history[phase]:
                self.metrics_history[phase][metric_name].append(value)

        if lr is not None:
            self.learning_rates.append(lr)

        # Log metrics
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} {phase.capitalize()} - {metric_str}")

    def add_epoch_time(self, epoch_time: float):
        """Add epoch training time."""
        self.epoch_times.append(epoch_time)

    def get_best_metric(self, phase: str, metric: str, mode: str = "min") -> tuple:
        """Get best metric value and epoch."""
        if (
            metric not in self.metrics_history[phase]
            or not self.metrics_history[phase][metric]
        ):
            return None, None

        values = self.metrics_history[phase][metric]
        if mode == "min":
            best_value = min(values)
            best_epoch = values.index(best_value)
        else:
            best_value = max(values)
            best_epoch = values.index(best_value)

        return best_value, best_epoch

    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save metrics to JSON file."""
        metrics_data = {
            "metrics_history": self.metrics_history,
            "epoch_times": self.epoch_times,
            "learning_rates": self.learning_rates,
            "summary": {
                "total_epochs": len(self.epoch_times),
                "total_time": sum(self.epoch_times),
                "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0,
                "best_val_loss": self.get_best_metric("val", "loss", "min")[0],
                "best_val_accuracy": self.get_best_metric("val", "accuracy", "max")[0],
            },
        }

        save_path = self.save_dir / filename
        with open(save_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        self.logger.info(f"Metrics saved to {save_path}")


class MedicalMetrics:
    """Medical imaging specific metrics calculator."""

    def __init__(self, num_classes: int, task_type: str = "classification"):
        self.num_classes = num_classes
        self.task_type = task_type
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_classification_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = torch.softmax(predictions, dim=1)
                pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                pred_probs = predictions.cpu().numpy()
            else:
                pred_classes = predictions.cpu().numpy()
                pred_probs = None
        else:
            pred_classes = predictions
            pred_probs = None

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # Basic metrics
        accuracy = accuracy_score(targets, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, pred_classes, average="weighted", zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        # AUC for binary classification
        if self.num_classes == 2 and pred_probs is not None:
            try:
                auc = roc_auc_score(targets, pred_probs[:, 1])
                metrics["auc"] = auc
            except ValueError:
                metrics["auc"] = 0.0

        return metrics

    def compute_dice_coefficient(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Compute Dice coefficient for segmentation."""
        smooth = 1e-6

        if predictions.dim() > 1 and predictions.size(1) > 1:
            predictions = F.softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)

        if targets.dim() == predictions.dim() + 1:
            targets = torch.argmax(targets, dim=1)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum().float()
        union = predictions.sum().float() + targets.sum().float()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()

    def compute_iou(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Intersection over Union for segmentation."""
        smooth = 1e-6

        if predictions.dim() > 1 and predictions.size(1) > 1:
            predictions = F.softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)

        if targets.dim() == predictions.dim() + 1:
            targets = torch.argmax(targets, dim=1)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum().float()
        union = predictions.sum().float() + targets.sum().float() - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

    def generate_confusion_matrix(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> np.ndarray:
        """Generate confusion matrix."""
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.cpu().numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        return confusion_matrix(targets, predictions, labels=range(self.num_classes))

    def compute_segmentation_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute comprehensive segmentation metrics."""
        dice = self.compute_dice_coefficient(predictions, targets)
        iou = self.compute_iou(predictions, targets)

        # Also compute classification metrics on flattened predictions
        if predictions.dim() > 1 and predictions.size(1) > 1:
            flat_preds = torch.argmax(predictions, dim=1).view(-1)
        else:
            flat_preds = predictions.view(-1)

        if targets.dim() == predictions.dim():
            flat_targets = torch.argmax(targets, dim=1).view(-1)
        else:
            flat_targets = targets.view(-1)

        cls_metrics = self.compute_classification_metrics(flat_preds, flat_targets)

        return {"dice_coefficient": dice, "iou_score": iou, **cls_metrics}


class CheckpointManager:
    """Manage model checkpoints and best model saving."""

    def __init__(
        self, checkpoint_dir: str = "outputs/checkpoints", save_frequency: int = 10
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency

        self.best_metrics = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False,
        filename: Optional[str] = None,
    ):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics,
            "config": config,
            "best_metrics": self.best_metrics,
        }

        # Save regular checkpoint
        if epoch % self.save_frequency == 0 or is_best:
            checkpoint_path = self.checkpoint_dir / filename
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scaler: torch.cuda.amp.GradScaler = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.best_metrics = checkpoint.get("best_metrics", {})

        self.logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")

        return checkpoint

    def update_best_metrics(self, metrics: Dict[str, float], epoch: int):
        """Update best metrics tracking."""
        for metric_name, value in metrics.items():
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = {"value": value, "epoch": epoch}
            else:
                # Assume lower is better for loss, higher is better for others
                if "loss" in metric_name.lower():
                    if value < self.best_metrics[metric_name]["value"]:
                        self.best_metrics[metric_name] = {
                            "value": value,
                            "epoch": epoch,
                        }
                else:
                    if value > self.best_metrics[metric_name]["value"]:
                        self.best_metrics[metric_name] = {
                            "value": value,
                            "epoch": epoch,
                        }

    def is_best_model(
        self, metrics: Dict[str, float], primary_metric: str = "val_loss"
    ) -> bool:
        """Check if current metrics represent the best model."""
        if primary_metric not in metrics:
            return False

        if primary_metric not in self.best_metrics:
            return True

        current_value = metrics[primary_metric]
        best_value = self.best_metrics[primary_metric]["value"]

        # Assume lower is better for loss, higher is better for others
        if "loss" in primary_metric.lower():
            return current_value < best_value
        else:
            return current_value > best_value


def create_metrics_tracker(config: Dict[str, Any]) -> MetricsTracker:
    """Factory function to create metrics tracker from configuration."""
    paths_config = config.get("paths", {})
    logs_dir = paths_config.get("logs_dir", "outputs/logs")

    return MetricsTracker(save_dir=logs_dir)


def create_checkpoint_manager(config: Dict[str, Any]) -> CheckpointManager:
    """Factory function to create checkpoint manager from configuration."""
    paths_config = config.get("paths", {})
    checkpoint_dir = paths_config.get("checkpoint_dir", "outputs/checkpoints")

    training_config = config.get("training", {})
    save_frequency = training_config.get("checkpoint_frequency", 10)

    return CheckpointManager(
        checkpoint_dir=checkpoint_dir, save_frequency=save_frequency
    )
