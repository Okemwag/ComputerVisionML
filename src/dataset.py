"""
Medical Image Dataset Module

PyTorch Dataset classes for medical image analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.augmentations import MedicalAugmentations

from .loaders import DataLoadingError, ImageLoader
from .preprocessing import MedicalImagePreprocessor


class MedicalImageDataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for medical images."""

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        transform: Optional[callable] = None,
        target_size: Tuple[int, int] = (256, 256),
        normalize_method: str = "hounsfield",
        window_type: str = "soft_tissue",
        use_dicom: bool = True,
        use_tiff: bool = True,
        split: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        self.data_dir = Path(data_dir)
        self.metadata_file = metadata_file
        self.transform = transform
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.window_type = window_type
        self.use_dicom = use_dicom
        self.use_tiff = use_tiff
        self.split = split
        self.split_ratios = split_ratios

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.image_loader = ImageLoader()
        self.preprocessor_config = {
            "image_size": list(target_size),
            "normalize_method": normalize_method,
            "window_type": window_type,
        }
        self.preprocessor = MedicalImagePreprocessor(self.preprocessor_config)

        # Load and prepare data
        self._load_metadata()
        self._prepare_dataset()
        self._create_splits()

        self.logger.info(f"Dataset initialized with {len(self.data_samples)} samples")
        if self.split:
            self.logger.info(f"Using {self.split} split with {len(self)} samples")

    def _load_metadata(self):
        """Load metadata from CSV file"""
        try:
            self.metadata_df = pd.read_csv(self.metadata_file)
            self.logger.info(f"Loaded metadata with {len(self.metadata_df)} entries")
        except Exception as e:
            raise DataLoadingError(
                f"Failed to load metadata from {self.metadata_file}: {str(e)}"
            )

    def _prepare_dataset(self):
        """Prepare dataset by validating files and creating sample list"""
        self.data_samples = []

        for idx, row in self.metadata_df.iterrows():
            sample = {
                "id": row["id"],
                "age": row["Age"],
                "contrast": bool(row["Contrast"]),
                "contrast_tag": row.get("ContrastTag", ""),
                "metadata": row.to_dict(),
            }

            # Add file paths based on preferences
            if self.use_dicom and "dicom_name" in row:
                dicom_path = self.data_dir / "dicom_dir" / row["dicom_name"]
                if dicom_path.exists():
                    sample["dicom_path"] = str(dicom_path)

            if self.use_tiff and "tiff_name" in row:
                tiff_path = self.data_dir / "tiff_images" / row["tiff_name"]
                if tiff_path.exists():
                    sample["tiff_path"] = str(tiff_path)

            # Only include samples with at least one valid file
            if "dicom_path" in sample or "tiff_path" in sample:
                self.data_samples.append(sample)

        self.logger.info(f"Prepared {len(self.data_samples)} valid samples")

    def _create_splits(self):
        """Create train/validation/test splits"""
        if self.split is None:
            self.split_indices = list(range(len(self.data_samples)))
            return

        # Extract labels for stratification
        labels = [sample["contrast"] for sample in self.data_samples]
        indices = list(range(len(self.data_samples)))

        # First split: train vs (val + test)
        train_ratio = self.split_ratios[0]
        val_test_ratio = 1 - train_ratio

        train_indices, val_test_indices = train_test_split(
            indices, test_size=val_test_ratio, stratify=labels, random_state=42
        )

        # Second split: val vs test
        val_ratio = self.split_ratios[1] / val_test_ratio
        val_test_labels = [labels[i] for i in val_test_indices]
        val_indices, test_indices = train_test_split(
            val_test_indices,
            test_size=(1 - val_ratio),
            stratify=val_test_labels,
            random_state=42,
        )

        # Store splits
        self.splits = {"train": train_indices, "val": val_indices, "test": test_indices}

        if self.split in self.splits:
            self.split_indices = self.splits[self.split]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        # Log split information
        for split_name, split_indices in self.splits.items():
            split_labels = [labels[i] for i in split_indices]
            contrast_count = sum(split_labels)
            no_contrast_count = len(split_labels) - contrast_count
            self.logger.info(
                f"{split_name.capitalize()} split: {len(split_indices)} samples "
                f"(Contrast: {contrast_count}, No-contrast: {no_contrast_count})"
            )

    def __len__(self) -> int:
        """Return dataset length"""
        if hasattr(self, "split_indices"):
            return len(self.split_indices)
        return len(self.data_samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Get a sample from the dataset."""
        # Get actual sample index if using splits
        if hasattr(self, "split_indices"):
            actual_idx = self.split_indices[idx]
        else:
            actual_idx = idx

        sample = self.data_samples[actual_idx]

        try:
            # Load image (prefer DICOM if available)
            if "dicom_path" in sample and self.use_dicom:
                image_path = sample["dicom_path"]
            elif "tiff_path" in sample and self.use_tiff:
                image_path = sample["tiff_path"]
            else:
                raise DataLoadingError(f"No valid image path for sample {sample['id']}")

            # Load and preprocess image
            medical_image = self.image_loader.load_image(image_path, sample["metadata"])
            processed_image = self.preprocessor.preprocess_medical_image(medical_image)

            # Get image tensor
            image_tensor = processed_image.image_data

            # Apply additional transforms if provided
            if self.transform:
                image_tensor = self.transform(image_tensor)

            # Prepare labels and metadata as tensors
            contrast_label = torch.tensor(int(sample["contrast"]), dtype=torch.long)
            age = torch.tensor(float(sample["age"]), dtype=torch.float32)
            sample_id = str(sample["id"])

            return image_tensor, contrast_label, age, sample_id

        except Exception as e:
            self.logger.error(f"Failed to load sample {sample['id']}: {str(e)}")
            # Return a dummy sample to avoid breaking the dataloader
            dummy_tensor = torch.zeros((1, *self.target_size), dtype=torch.float32)
            dummy_label = torch.tensor(int(sample["contrast"]), dtype=torch.long)
            dummy_age = torch.tensor(float(sample["age"]), dtype=torch.float32)
            dummy_id = str(sample["id"])

            return dummy_tensor, dummy_label, dummy_age, dummy_id

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        if hasattr(self, "split_indices"):
            labels = [self.data_samples[i]["contrast"] for i in self.split_indices]
        else:
            labels = [sample["contrast"] for sample in self.data_samples]

        contrast_count = sum(labels)
        no_contrast_count = len(labels) - contrast_count
        total_count = len(labels)

        if contrast_count == 0 or no_contrast_count == 0:
            return torch.tensor([1.0, 1.0])

        no_contrast_weight = total_count / (2 * no_contrast_count)
        contrast_weight = total_count / (2 * contrast_count)

        weights = torch.tensor([no_contrast_weight, contrast_weight])
        self.logger.info(
            f"Class weights: No-contrast={no_contrast_weight:.3f}, Contrast={contrast_weight:.3f}"
        )

        return weights

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {
            "total_samples": len(self.data_samples),
            "current_split_samples": len(self),
            "split_ratios": self.split_ratios,
            "target_size": self.target_size,
            "normalize_method": self.normalize_method,
        }

        # Class distribution
        if hasattr(self, "split_indices"):
            labels = [self.data_samples[i]["contrast"] for i in self.split_indices]
        else:
            labels = [sample["contrast"] for sample in self.data_samples]

        contrast_count = sum(labels)
        no_contrast_count = len(labels) - contrast_count

        stats["class_distribution"] = {
            "contrast": contrast_count,
            "no_contrast": no_contrast_count,
            "contrast_ratio": contrast_count / len(labels) if labels else 0,
        }

        # Age statistics
        ages = [sample["age"] for sample in self.data_samples]
        if ages:
            stats["age_statistics"] = {
                "mean": np.mean(ages),
                "std": np.std(ages),
                "min": min(ages),
                "max": max(ages),
                "median": np.median(ages),
            }

        return stats

    @classmethod
    def create_datasets(
        cls, data_dir: str, metadata_file: str, **kwargs
    ) -> Tuple["MedicalImageDataset", "MedicalImageDataset", "MedicalImageDataset"]:
        """Create train, validation, and test datasets."""
        train_dataset = cls(data_dir, metadata_file, split="train", **kwargs)
        val_dataset = cls(data_dir, metadata_file, split="val", **kwargs)
        test_dataset = cls(data_dir, metadata_file, split="test", **kwargs)

        return train_dataset, val_dataset, test_dataset


class AugmentedMedicalDataset(MedicalImageDataset):
    """Extended MedicalImageDataset with augmentation support."""

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        augmentation_config: Optional[Dict[str, Any]] = None,
        apply_augmentation: bool = True,
        **kwargs,
    ):
        super().__init__(data_dir, metadata_file, **kwargs)

        self.apply_augmentation = apply_augmentation

        # Set up augmentations
        if augmentation_config is None:
            augmentation_config = {
                "rotation_limit": 15,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "horizontal_flip": True,
                "vertical_flip": False,
                "gaussian_noise": 0.01,
                "augmentation_probability": 0.8,
            }

        if self.apply_augmentation:
            self.augmentations = MedicalAugmentations(**augmentation_config)
            self.logger.info("Augmentations enabled for this dataset")
        else:
            self.augmentations = None
            self.logger.info("Augmentations disabled for this dataset")

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Get augmented sample from dataset."""
        # Get base sample
        image_tensor, contrast_label, age, sample_id = super().__getitem__(idx)

        # Apply augmentations if enabled
        if self.apply_augmentation and self.augmentations is not None:
            image_tensor = self.augmentations(image_tensor)

        return image_tensor, contrast_label, age, sample_id

    @classmethod
    def create_augmented_datasets(
        cls,
        data_dir: str,
        metadata_file: str,
        augmentation_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[
        "AugmentedMedicalDataset", "AugmentedMedicalDataset", "AugmentedMedicalDataset"
    ]:
        """Create train (augmented), validation, and test datasets."""
        train_dataset = cls(
            data_dir,
            metadata_file,
            split="train",
            augmentation_config=augmentation_config,
            apply_augmentation=True,
            **kwargs,
        )

        val_dataset = cls(
            data_dir,
            metadata_file,
            split="val",
            augmentation_config=None,
            apply_augmentation=False,
            **kwargs,
        )

        test_dataset = cls(
            data_dir,
            metadata_file,
            split="test",
            augmentation_config=None,
            apply_augmentation=False,
            **kwargs,
        )

        return train_dataset, val_dataset, test_dataset
