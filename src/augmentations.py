"""
Medical Image Augmentation Module

Provides medical-specific data augmentation pipelines.
"""

import logging
from typing import Any, Dict

import torch


class MedicalAugmentations:
    """Medical-specific data augmentation pipeline."""
    
    def __init__(
        self,
        rotation_limit: int = 15,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        gaussian_noise: float = 0.01,
        augmentation_probability: float = 0.8
    ):
        self.rotation_limit = rotation_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.gaussian_noise = gaussian_noise
        self.augmentation_probability = augmentation_probability
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self):
        """Create albumentations augmentation pipeline"""
        try:
            import albumentations as A
            
            transforms = []
            
            if self.rotation_limit > 0:
                transforms.append(A.Rotate(limit=self.rotation_limit, p=0.5))
            
            if self.horizontal_flip:
                transforms.append(A.HorizontalFlip(p=0.5))
            
            if self.vertical_flip:
                transforms.append(A.VerticalFlip(p=0.3))
            
            if self.brightness_limit > 0 or self.contrast_limit > 0:
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=self.brightness_limit,
                        contrast_limit=self.contrast_limit,
                        p=0.6
                    )
                )
            
            if self.gaussian_noise > 0:
                transforms.append(A.GaussNoise(var_limit=(0, self.gaussian_noise), p=0.3))
            
            transforms.extend([
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ])
            
            self.augmentation_pipeline = A.Compose(transforms, p=self.augmentation_probability)
            
        except ImportError:
            self._create_torch_pipeline()
    
    def _create_torch_pipeline(self):
        """Create basic PyTorch transforms as fallback"""
        import torchvision.transforms as T
        
        transforms = []
        
        if self.rotation_limit > 0:
            transforms.append(T.RandomRotation(degrees=self.rotation_limit))
        
        if self.horizontal_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        
        if self.vertical_flip:
            transforms.append(T.RandomVerticalFlip(p=0.3))
        
        if self.brightness_limit > 0 or self.contrast_limit > 0:
            transforms.append(
                T.ColorJitter(
                    brightness=self.brightness_limit,
                    contrast=self.contrast_limit
                )
            )
        
        self.torch_pipeline = T.Compose(transforms)
        self.use_torch = True
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to image tensor."""
        if hasattr(self, 'use_torch') and self.use_torch:
            return self._apply_torch_transforms(image)
        else:
            return self._apply_albumentations(image)
    
    def _apply_albumentations(self, image: torch.Tensor) -> torch.Tensor:
        """Apply albumentations transforms"""
        if image.dim() == 3 and image.shape[0] == 1:
            image_np = image.squeeze(0).numpy()
        else:
            image_np = image.permute(1, 2, 0).numpy()
        
        augmented = self.augmentation_pipeline(image=image_np)
        augmented_image = augmented['image']
        
        if len(augmented_image.shape) == 2:
            result = torch.from_numpy(augmented_image).unsqueeze(0)
        else:
            result = torch.from_numpy(augmented_image).permute(2, 0, 1)
        
        return result.float()
    
    def _apply_torch_transforms(self, image: torch.Tensor) -> torch.Tensor:
        """Apply PyTorch transforms"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        augmented = self.torch_pipeline(image)
        
        if augmented.shape[0] == 1:
            augmented = augmented.squeeze(0)
        
        return augmented


def create_medical_transforms(config: Dict[str, Any]) -> MedicalAugmentations:
    """Factory function to create medical augmentations from configuration."""
    augmentation_config = config.get('augmentation', {})
    
    return MedicalAugmentations(
        rotation_limit=augmentation_config.get('rotation_limit', 15),
        brightness_limit=augmentation_config.get('brightness_limit', 0.2),
        contrast_limit=augmentation_config.get('contrast_limit', 0.2),
        horizontal_flip=augmentation_config.get('horizontal_flip', True),
        vertical_flip=augmentation_config.get('vertical_flip', False),
        gaussian_noise=augmentation_config.get('gaussian_noise', 0.01),
        augmentation_probability=augmentation_config.get('augmentation_probability', 0.8)
    )