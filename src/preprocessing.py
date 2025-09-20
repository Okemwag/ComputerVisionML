"""
Medical Image Preprocessing Module

Handles image preprocessing, normalization, and augmentation for medical images.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .loaders import DataLoadingError, MedicalImage


class ImagePreprocessor:
    """Image preprocessing class for medical images."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256), normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # CT scan Hounsfield unit ranges
        self.hu_min = -1024
        self.hu_max = 3071
        self.soft_tissue_window = (-160, 240)
        self.bone_window = (-450, 1050)
        self.lung_window = (-1200, 600)
    
    def _preprocess_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing pipeline to image tensor."""
        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
        
        if self.target_size != image_tensor.shape[-2:]:
            image_tensor = self._resize_tensor(image_tensor, self.target_size)
        
        if self.normalize:
            image_tensor = self.normalize_hounsfield(image_tensor)
        
        return image_tensor
    
    def _resize_tensor(self, tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Resize tensor while preserving aspect ratio."""
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        
        if resized.shape[0] == 1:
            resized = resized.squeeze(0)
        
        return resized
    
    def normalize_hounsfield(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize Hounsfield units to [0, 1] range."""
        clipped = torch.clamp(image_tensor, self.hu_min, self.hu_max)
        normalized = (clipped - self.hu_min) / (self.hu_max - self.hu_min)
        return normalized
    
    def normalize_to_window(self, image_tensor: torch.Tensor, window_type: str = 'soft_tissue') -> torch.Tensor:
        """Normalize image using specific CT window settings."""
        if window_type == 'soft_tissue':
            window_min, window_max = self.soft_tissue_window
        elif window_type == 'bone':
            window_min, window_max = self.bone_window
        elif window_type == 'lung':
            window_min, window_max = self.lung_window
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        
        windowed = torch.clamp(image_tensor, window_min, window_max)
        normalized = (windowed - window_min) / (window_max - window_min)
        return normalized


class MedicalImagePreprocessor:
    """Advanced preprocessor for medical images with multiple preprocessing options."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = config.get('image_size', [256, 256])
        self.normalize_method = config.get('normalize_method', 'hounsfield')
        self.window_type = config.get('window_type', 'soft_tissue')
        
        self.preprocessor = ImagePreprocessor(target_size=tuple(self.image_size), normalize=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def preprocess_medical_image(self, medical_image: MedicalImage) -> MedicalImage:
        """Preprocess a MedicalImage object."""
        try:
            if medical_image.image_format == 'dicom':
                processed_tensor = self._preprocess_with_metadata(
                    medical_image.image_data, medical_image.metadata
                )
            else:
                processed_tensor = self.preprocessor._preprocess_tensor(medical_image.image_data)
            
            processed_image = MedicalImage(
                image_id=medical_image.image_id,
                image_data=processed_tensor,
                metadata=medical_image.metadata,
                contrast_label=medical_image.contrast_label,
                age=medical_image.age,
                file_path=medical_image.file_path,
                image_format=medical_image.image_format
            )
            
            processed_image.metadata['preprocessing'] = {
                'target_size': self.image_size,
                'normalize_method': self.normalize_method,
                'original_shape': tuple(medical_image.image_data.shape)
            }
            
            return processed_image
            
        except Exception as e:
            raise DataLoadingError(f"Failed to preprocess medical image {medical_image.image_id}: {str(e)}")
    
    def _preprocess_with_metadata(self, image_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Preprocess image tensor using metadata information."""
        if 'rescale_slope' in metadata and 'rescale_intercept' in metadata:
            slope = float(metadata['rescale_slope'])
            intercept = float(metadata['rescale_intercept'])
            image_tensor = image_tensor * slope + intercept
        
        if self.normalize_method == 'window':
            image_tensor = self.preprocessor.normalize_to_window(image_tensor, self.window_type)
        elif self.normalize_method == 'hounsfield':
            image_tensor = self.preprocessor.normalize_hounsfield(image_tensor)
        elif self.normalize_method == 'zscore':
            mean = torch.mean(image_tensor)
            std = torch.std(image_tensor)
            image_tensor = (image_tensor - mean) / (std + 1e-8)
        elif self.normalize_method == 'minmax':
            min_val = torch.min(image_tensor)
            max_val = torch.max(image_tensor)
            image_tensor = (image_tensor - min_val) / (max_val - min_val + 1e-8)
        
        if tuple(self.image_size) != image_tensor.shape[-2:]:
            image_tensor = self.preprocessor._resize_tensor(image_tensor, tuple(self.image_size))
        
        return image_tensor