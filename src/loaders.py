"""
Medical Image Loaders Module

Handles loading DICOM and TIFF medical images with metadata extraction.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from PIL import Image
from pydicom.errors import InvalidDicomError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoadingError(Exception):
    """Custom exception for data loading errors"""
    pass


@dataclass
class MedicalImage:
    """
    Standardized medical image data structure.
    
    Attributes:
        image_id: Unique identifier for the image
        image_data: Image tensor with shape (C, H, W)
        metadata: Dictionary containing image metadata
        contrast_label: Boolean indicating contrast enhancement
        age: Patient age
        file_path: Path to the original image file
        image_format: Format of the image ('dicom' or 'tiff')
    """
    image_id: str
    image_data: torch.Tensor
    metadata: Dict[str, Any]
    contrast_label: bool
    age: int
    file_path: str
    image_format: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'image_id': self.image_id,
            'metadata': self.metadata,
            'contrast_label': self.contrast_label,
            'age': self.age,
            'file_path': self.file_path,
            'image_format': self.image_format,
            'image_shape': self.image_data.shape
        }
    
    @classmethod
    def from_dicom(cls, dicom_path: str, metadata: Dict[str, Any]) -> 'MedicalImage':
        """Create MedicalImage instance from DICOM file"""
        try:
            dicom_loader = DICOMLoader()
            image_data, dicom_metadata = dicom_loader.load_dicom(dicom_path)
            
            image_id = metadata.get('id', Path(dicom_path).stem)
            contrast_label = metadata.get('Contrast', False)
            age = metadata.get('Age', 0)
            
            combined_metadata = {**metadata, **dicom_metadata}
            
            return cls(
                image_id=str(image_id),
                image_data=image_data,
                metadata=combined_metadata,
                contrast_label=bool(contrast_label),
                age=int(age),
                file_path=dicom_path,
                image_format='dicom'
            )
        except Exception as e:
            raise DataLoadingError(f"Failed to create MedicalImage from DICOM {dicom_path}: {str(e)}")


class DICOMLoader:
    """Specialized loader for DICOM files with metadata extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_dicom(self, dicom_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load DICOM file and extract image data and metadata."""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            pixel_array = dicom_data.pixel_array
            
            if pixel_array.dtype != np.float32:
                pixel_array = pixel_array.astype(np.float32)
            
            if len(pixel_array.shape) == 2:
                pixel_array = np.expand_dims(pixel_array, axis=0)
            elif len(pixel_array.shape) == 3:
                if pixel_array.shape[2] == 3:
                    pixel_array = np.transpose(pixel_array, (2, 0, 1))
            
            image_tensor = torch.from_numpy(pixel_array)
            metadata = self._extract_dicom_metadata(dicom_data)
            
            return image_tensor, metadata
            
        except InvalidDicomError as e:
            raise DataLoadingError(f"Invalid DICOM file {dicom_path}: {str(e)}")
        except Exception as e:
            raise DataLoadingError(f"Error loading DICOM file {dicom_path}: {str(e)}")
    
    def _extract_dicom_metadata(self, dicom_data) -> Dict[str, Any]:
        """Extract relevant metadata from DICOM dataset."""
        metadata = {}
        
        metadata_fields = {
            'PatientID': 'patient_id',
            'PatientAge': 'patient_age',
            'PatientSex': 'patient_sex',
            'StudyDate': 'study_date',
            'StudyTime': 'study_time',
            'Modality': 'modality',
            'SliceThickness': 'slice_thickness',
            'PixelSpacing': 'pixel_spacing',
            'WindowCenter': 'window_center',
            'WindowWidth': 'window_width',
            'RescaleIntercept': 'rescale_intercept',
            'RescaleSlope': 'rescale_slope',
            'ContrastBolusAgent': 'contrast_agent',
            'KVP': 'kvp',
            'XRayTubeCurrent': 'tube_current'
        }
        
        for dicom_tag, metadata_key in metadata_fields.items():
            try:
                if hasattr(dicom_data, dicom_tag):
                    value = getattr(dicom_data, dicom_tag)
                    metadata[metadata_key] = value
            except Exception:
                pass
        
        metadata['image_height'] = dicom_data.Rows if hasattr(dicom_data, 'Rows') else None
        metadata['image_width'] = dicom_data.Columns if hasattr(dicom_data, 'Columns') else None
        metadata['bits_allocated'] = dicom_data.BitsAllocated if hasattr(dicom_data, 'BitsAllocated') else None
        
        return metadata


class TIFFLoader:
    """Loader for TIFF format medical images."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_tiff(self, tiff_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load TIFF file and extract image data."""
        try:
            pixel_array = None
            
            # Try PIL first
            try:
                with Image.open(tiff_path) as image:
                    pixel_array = np.array(image)
            except Exception:
                pass
            
            # Try SimpleITK with different pixel types
            if pixel_array is None:
                try:
                    sitk_image = sitk.ReadImage(tiff_path)
                    pixel_array = sitk.GetArrayFromImage(sitk_image)
                except Exception:
                    try:
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(tiff_path)
                        reader.ReadImageInformation()
                        
                        if reader.GetPixelID() in [sitk.sitkFloat64, sitk.sitkInt64, sitk.sitkUInt64]:
                            reader.SetOutputPixelType(sitk.sitkFloat32)
                        
                        sitk_image = reader.Execute()
                        pixel_array = sitk.GetArrayFromImage(sitk_image)
                    except Exception:
                        pass
            
            # Try tifffile as last resort
            if pixel_array is None:
                try:
                    import tifffile
                    pixel_array = tifffile.imread(tiff_path)
                except Exception:
                    pass
            
            if pixel_array is None:
                raise DataLoadingError(f"Could not load TIFF file: {tiff_path}")
            
            if pixel_array.size == 0:
                raise DataLoadingError(f"Loaded empty array from TIFF file: {tiff_path}")
            
            if pixel_array.dtype != np.float32:
                pixel_array = pixel_array.astype(np.float32)
            
            if len(pixel_array.shape) == 2:
                pixel_array = np.expand_dims(pixel_array, axis=0)
            elif len(pixel_array.shape) == 3:
                if pixel_array.shape[2] == 3:
                    pixel_array = np.transpose(pixel_array, (2, 0, 1))
                elif pixel_array.shape[0] == 3:
                    pass
                else:
                    pixel_array = pixel_array.squeeze()
                    if len(pixel_array.shape) == 2:
                        pixel_array = np.expand_dims(pixel_array, axis=0)
            
            image_tensor = torch.from_numpy(pixel_array)
            metadata = self._extract_tiff_metadata(tiff_path, image_tensor.shape)
            
            return image_tensor, metadata
            
        except Exception as e:
            raise DataLoadingError(f"Error loading TIFF file {tiff_path}: {str(e)}")
    
    def _extract_tiff_metadata(self, tiff_path: str, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Extract basic metadata from TIFF file."""
        metadata = {
            'file_path': tiff_path,
            'file_name': Path(tiff_path).name,
            'image_shape': image_shape,
            'channels': image_shape[0] if len(image_shape) >= 3 else 1,
            'height': image_shape[-2] if len(image_shape) >= 2 else None,
            'width': image_shape[-1] if len(image_shape) >= 1 else None,
        }
        
        filename = Path(tiff_path).stem
        if 'AGE_' in filename:
            try:
                age_part = filename.split('AGE_')[1].split('_')[0]
                metadata['filename_age'] = int(age_part)
            except (IndexError, ValueError):
                pass
        
        if 'CONTRAST_' in filename:
            try:
                contrast_part = filename.split('CONTRAST_')[1].split('_')[0]
                metadata['filename_contrast'] = int(contrast_part)
            except (IndexError, ValueError):
                pass
        
        return metadata


class ImageLoader:
    """Unified image loader that handles both DICOM and TIFF formats."""
    
    def __init__(self):
        self.dicom_loader = DICOMLoader()
        self.tiff_loader = TIFFLoader()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> MedicalImage:
        """Load medical image from file path."""
        if metadata is None:
            metadata = {}
        
        file_path = Path(image_path)
        
        if not file_path.exists():
            raise DataLoadingError(f"Image file not found: {image_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.dcm':
                return MedicalImage.from_dicom(str(file_path), metadata)
            elif file_extension in ['.tif', '.tiff']:
                return self._create_medical_image_from_tiff(str(file_path), metadata)
            else:
                try:
                    return MedicalImage.from_dicom(str(file_path), metadata)
                except:
                    return self._create_medical_image_from_tiff(str(file_path), metadata)
                    
        except Exception as e:
            raise DataLoadingError(f"Failed to load image {image_path}: {str(e)}")
    
    def _create_medical_image_from_tiff(self, tiff_path: str, metadata: Dict[str, Any]) -> MedicalImage:
        """Create MedicalImage instance from TIFF file"""
        try:
            image_data, tiff_metadata = self.tiff_loader.load_tiff(tiff_path)
            
            image_id = metadata.get('id', Path(tiff_path).stem)
            contrast_label = metadata.get('Contrast', False)
            age = metadata.get('Age', 0)
            
            combined_metadata = {**metadata, **tiff_metadata}
            
            return MedicalImage(
                image_id=str(image_id),
                image_data=image_data,
                metadata=combined_metadata,
                contrast_label=bool(contrast_label),
                age=int(age),
                file_path=tiff_path,
                image_format='tiff'
            )
        except Exception as e:
            raise DataLoadingError(f"Failed to create MedicalImage from TIFF {tiff_path}: {str(e)}")
    
    def load_batch(self, image_paths: list, metadata_list: Optional[list] = None) -> list:
        """Load multiple images in batch."""
        if metadata_list is None:
            metadata_list = [{}] * len(image_paths)
        
        medical_images = []
        failed_loads = []
        
        for i, (image_path, metadata) in enumerate(zip(image_paths, metadata_list)):
            try:
                medical_image = self.load_image(image_path, metadata)
                medical_images.append(medical_image)
            except DataLoadingError as e:
                self.logger.warning(f"Failed to load image {image_path}: {str(e)}")
                failed_loads.append((i, image_path, str(e)))
        
        if failed_loads:
            self.logger.info(f"Successfully loaded {len(medical_images)}/{len(image_paths)} images")
        
        return medical_images