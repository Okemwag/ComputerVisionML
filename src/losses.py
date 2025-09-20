"""
Medical Image Analysis - Loss Functions

Specialized loss functions for medical image classification and segmentation.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply softmax to predictions if not already applied
        if predictions.dim() > 1 and predictions.size(1) > 1:
            predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot if needed
        if targets.dim() == predictions.dim() - 1:
            targets = F.one_hot(targets, num_classes=predictions.size(1)).permute(0, -1, *range(1, targets.dim())).float()
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            dice = dice[:, [i for i in range(dice.size(1)) if i != self.ignore_index]]
        
        return 1.0 - dice.mean()


class IoULoss(nn.Module):
    """Intersection over Union (IoU) loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply softmax to predictions if not already applied
        if predictions.dim() > 1 and predictions.size(1) > 1:
            predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot if needed
        if targets.dim() == predictions.dim() - 1:
            targets = F.one_hot(targets, num_classes=predictions.size(1)).permute(0, -1, *range(1, targets.dim())).float()
        
        # Calculate IoU
        intersection = (predictions * targets).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            iou = iou[:, [i for i in range(iou.size(1)) if i != self.ignore_index]]
        
        return 1.0 - iou.mean()


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight
        
        # Classification loss
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Segmentation losses
        self.seg_ce_criterion = nn.CrossEntropyLoss()
        self.seg_dice_criterion = DiceLoss()
    
    def forward(
        self,
        cls_predictions: torch.Tensor,
        seg_predictions: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        # Classification loss
        cls_loss = self.cls_criterion(cls_predictions, cls_targets)
        
        # Segmentation loss (combination of CE and Dice)
        seg_ce_loss = self.seg_ce_criterion(seg_predictions, seg_targets)
        seg_dice_loss = self.seg_dice_criterion(seg_predictions, seg_targets)
        seg_loss = seg_ce_loss + seg_dice_loss
        
        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss +
            self.segmentation_weight * seg_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'segmentation_loss': seg_loss,
            'segmentation_ce_loss': seg_ce_loss,
            'segmentation_dice_loss': seg_dice_loss
        }


class LossManager:
    """Manager for different loss functions based on task type."""
    
    def __init__(self, task_type: str, class_weights: Optional[torch.Tensor] = None, config: Optional[Dict[str, Any]] = None):
        self.task_type = task_type
        self.class_weights = class_weights
        self.config = config or {}
        
        self._setup_loss_functions()
    
    def _setup_loss_functions(self):
        """Setup loss functions based on task type and configuration."""
        loss_config = self.config.get('loss', {})
        
        if self.task_type == 'classification':
            loss_type = loss_config.get('classification_loss', 'cross_entropy')
            
            if loss_type == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            elif loss_type == 'focal':
                alpha = loss_config.get('focal_alpha', 1.0)
                gamma = loss_config.get('focal_gamma', 2.0)
                self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
            else:
                raise ValueError(f"Unknown classification loss: {loss_type}")
                
        elif self.task_type == 'segmentation':
            loss_type = loss_config.get('segmentation_loss', 'dice')
            
            if loss_type == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            elif loss_type == 'dice':
                self.criterion = DiceLoss()
            elif loss_type == 'iou':
                self.criterion = IoULoss()
            elif loss_type == 'combined':
                # Combination of CE and Dice
                self.ce_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                self.dice_criterion = DiceLoss()
                self.criterion = self._combined_segmentation_loss
            else:
                raise ValueError(f"Unknown segmentation loss: {loss_type}")
                
        else:  # Multi-task
            loss_weights = loss_config.get('loss_weights', {'classification': 1.0, 'segmentation': 1.0})
            self.criterion = CombinedLoss(
                classification_weight=loss_weights['classification'],
                segmentation_weight=loss_weights['segmentation'],
                class_weights=self.class_weights
            )
    
    def _combined_segmentation_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined CE and Dice loss for segmentation."""
        ce_loss = self.ce_criterion(predictions, targets)
        dice_loss = self.dice_criterion(predictions, targets)
        return ce_loss + dice_loss
    
    def compute_loss(self, predictions, targets, **kwargs) -> torch.Tensor:
        """Compute loss based on task type."""
        if self.task_type in ['classification', 'segmentation']:
            return self.criterion(predictions, targets)
        else:  # Multi-task
            cls_predictions = kwargs.get('cls_predictions')
            seg_predictions = kwargs.get('seg_predictions')
            cls_targets = kwargs.get('cls_targets')
            seg_targets = kwargs.get('seg_targets')
            
            return self.criterion(cls_predictions, seg_predictions, cls_targets, seg_targets)


def create_loss_function(task_type: str, config: Dict[str, Any], class_weights: Optional[torch.Tensor] = None):
    """
    Factory function to create loss function from configuration.
    
    Args:
        task_type: Type of task ('classification', 'segmentation', 'both')
        config: Configuration dictionary
        class_weights: Optional class weights for imbalanced datasets
        
    Returns:
        Loss function or LossManager instance
    """
    return LossManager(task_type, class_weights, config)


# Utility functions for loss computation
def compute_dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute Dice coefficient for evaluation."""
    if predictions.dim() > 1 and predictions.size(1) > 1:
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
    
    if targets.dim() == predictions.dim() + 1:
        targets = torch.argmax(targets, dim=1)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def compute_iou(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute IoU for evaluation."""
    if predictions.dim() > 1 and predictions.size(1) > 1:
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
    
    if targets.dim() == predictions.dim() + 1:
        targets = torch.argmax(targets, dim=1)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou