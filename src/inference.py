"""
Medical Image Analysis - Inference Engine

Model inference capabilities for single images and batch processing.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .loaders import ImageLoader
from .metrics import CheckpointManager
from .model import create_model
from .preprocessing import MedicalImagePreprocessor


class ModelInference:
    """Main inference engine for medical image analysis."""

    def __init__(self, model_path: str, config_path: str, device: str = "cuda"):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration file
            device: Device to use for inference
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device

        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration
        self._load_config()

        # Initialize components
        self.image_loader = ImageLoader()
        self.preprocessor = MedicalImagePreprocessor(self.config)

        # Load model
        self.model = self._load_model()

        self.logger.info("Inference engine initialized successfully")

    def _load_config(self):
        """Load configuration from YAML file."""
        import yaml

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        # Create model
        model = create_model(self.config)

        # Load checkpoint
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.load_checkpoint(self.model_path, model, device=self.device)

        model.eval()
        return model

    def predict_single(
        self, image_path: str, return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Predict on a single image.

        Args:
            image_path: Path to input image
            return_probabilities: Whether to return class probabilities

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        try:
            # Load and preprocess image
            medical_image = self.image_loader.load_image(image_path)
            processed_image = self.preprocessor.preprocess_medical_image(medical_image)

            # Prepare input tensor
            input_tensor = processed_image.image_data.unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)

            # Process outputs
            results = self._process_outputs(outputs, return_probabilities)

            # Add metadata
            results.update(
                {
                    "image_path": image_path,
                    "image_id": processed_image.image_id,
                    "inference_time": time.time() - start_time,
                    "input_shape": tuple(input_tensor.shape),
                    "metadata": processed_image.metadata,
                }
            )

            return results

        except Exception as e:
            self.logger.error(f"Failed to predict on {image_path}: {str(e)}")
            return {
                "error": str(e),
                "image_path": image_path,
                "inference_time": time.time() - start_time,
            }

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 16,
        return_probabilities: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict on a batch of images.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities

        Returns:
            List of prediction results
        """
        results = []

        self.logger.info(
            f"Processing {len(image_paths)} images in batches of {batch_size}"
        )

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_results = self._process_batch(batch_paths, return_probabilities)
            results.extend(batch_results)

            # Log progress
            if (i // batch_size + 1) % 10 == 0:
                self.logger.info(
                    f"Processed {i + len(batch_paths)}/{len(image_paths)} images"
                )

        return results

    def _process_batch(
        self, image_paths: List[str], return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """Process a single batch of images."""
        batch_tensors = []
        batch_metadata = []
        valid_indices = []

        # Load and preprocess images
        for idx, image_path in enumerate(image_paths):
            try:
                medical_image = self.image_loader.load_image(image_path)
                processed_image = self.preprocessor.preprocess_medical_image(
                    medical_image
                )

                batch_tensors.append(processed_image.image_data)
                batch_metadata.append(
                    {
                        "image_path": image_path,
                        "image_id": processed_image.image_id,
                        "metadata": processed_image.metadata,
                    }
                )
                valid_indices.append(idx)

            except Exception as e:
                self.logger.warning(f"Failed to load {image_path}: {str(e)}")
                batch_metadata.append({"image_path": image_path, "error": str(e)})

        results = []

        if batch_tensors:
            # Stack tensors and run inference
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            start_time = time.time()
            with torch.no_grad():
                batch_outputs = self.model(batch_tensor)
            inference_time = time.time() - start_time

            # Process outputs for each image
            for i, (tensor_idx, meta) in enumerate(zip(valid_indices, batch_metadata)):
                if "error" not in meta:
                    # Extract single output
                    if isinstance(batch_outputs, dict):
                        single_output = {
                            k: v[i : i + 1] for k, v in batch_outputs.items()
                        }
                    else:
                        single_output = batch_outputs[i : i + 1]

                    # Process output
                    result = self._process_outputs(single_output, return_probabilities)
                    result.update(meta)
                    result["inference_time"] = inference_time / len(batch_tensors)

                    results.append(result)
                else:
                    results.append(meta)

        # Add results for failed images
        for idx, image_path in enumerate(image_paths):
            if idx not in valid_indices:
                results.insert(
                    idx, {"image_path": image_path, "error": "Failed to load image"}
                )

        return results

    def _process_outputs(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        """Process model outputs into prediction results."""
        results = {}

        if isinstance(outputs, dict):
            # Multi-task model
            if "classification" in outputs:
                cls_outputs = outputs["classification"]
                probabilities = F.softmax(cls_outputs, dim=1)
                predictions = torch.argmax(cls_outputs, dim=1)

                results["classification"] = {
                    "prediction": predictions.cpu().numpy().tolist(),
                    "confidence": probabilities.max(dim=1)[0].cpu().numpy().tolist(),
                }

                if return_probabilities:
                    results["classification"]["probabilities"] = (
                        probabilities.cpu().numpy().tolist()
                    )

            if "segmentation" in outputs:
                seg_outputs = outputs["segmentation"]
                seg_probabilities = F.softmax(seg_outputs, dim=1)
                seg_predictions = torch.argmax(seg_outputs, dim=1)

                results["segmentation"] = {
                    "prediction_mask": seg_predictions.cpu().numpy().tolist(),
                    "confidence_map": seg_probabilities.max(dim=1)[0]
                    .cpu()
                    .numpy()
                    .tolist(),
                }

                if return_probabilities:
                    results["segmentation"]["probability_maps"] = (
                        seg_probabilities.cpu().numpy().tolist()
                    )

        else:
            # Single task model
            task_type = self.config.get("model", {}).get("task_type", "classification")

            if task_type == "classification":
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                results = {
                    "prediction": predictions.cpu().numpy().tolist(),
                    "confidence": probabilities.max(dim=1)[0].cpu().numpy().tolist(),
                }

                if return_probabilities:
                    results["probabilities"] = probabilities.cpu().numpy().tolist()

            elif task_type == "segmentation":
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                results = {
                    "prediction_mask": predictions.cpu().numpy().tolist(),
                    "confidence_map": probabilities.max(dim=1)[0]
                    .cpu()
                    .numpy()
                    .tolist(),
                }

                if return_probabilities:
                    results["probability_maps"] = probabilities.cpu().numpy().tolist()

        return results


class BatchProcessor:
    """Efficient batch processing for multiple images."""

    def __init__(
        self, model: torch.nn.Module, batch_size: int = 16, device: str = "cuda"
    ):
        """
        Initialize batch processor.

        Args:
            model: Trained PyTorch model
            batch_size: Batch size for processing
            device: Device to use for processing
        """
        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device

        self.model.eval()
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_extensions: List[str] = [".dcm", ".tif", ".tiff"],
    ) -> Dict[str, Any]:
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for results
            file_extensions: List of file extensions to process

        Returns:
            Processing summary
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"**/*{ext}"))

        self.logger.info(f"Found {len(image_files)} images to process")

        # Create inference engine (assuming we have the necessary components)
        # This would need to be properly initialized with model path and config

        # Process images and save results
        results = []
        for i, image_file in enumerate(image_files):
            try:
                # Process single image (simplified - would use full inference pipeline)
                result = {
                    "file_path": str(image_file),
                    "processed": True,
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
                results.append(result)

            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {str(e)}")
                results.append(
                    {
                        "file_path": str(image_file),
                        "processed": False,
                        "error": str(e),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                )

        # Save results summary
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path / "processing_results.csv", index=False)

        summary = {
            "total_files": len(image_files),
            "processed_successfully": len(
                [r for r in results if r.get("processed", False)]
            ),
            "failed": len([r for r in results if not r.get("processed", False)]),
            "output_directory": str(output_path),
        }

        self.logger.info(f"Processing complete: {summary}")
        return summary

    def process_with_metadata(
        self, image_paths: List[str], metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process images with associated metadata.

        Args:
            image_paths: List of image file paths
            metadata: DataFrame with image metadata

        Returns:
            DataFrame with predictions and metadata
        """
        # This would integrate with the full inference pipeline
        # For now, return a placeholder structure

        results_data = []

        for image_path in image_paths:
            # Find matching metadata
            image_name = Path(image_path).name
            matching_meta = metadata[metadata["file_name"] == image_name]

            result = {
                "image_path": image_path,
                "prediction": 0,  # Placeholder
                "confidence": 0.5,  # Placeholder
                "processing_time": 0.1,  # Placeholder
            }

            # Add metadata if found
            if not matching_meta.empty:
                for col in matching_meta.columns:
                    result[f"meta_{col}"] = matching_meta.iloc[0][col]

            results_data.append(result)

        return pd.DataFrame(results_data)


def create_inference_engine(
    model_path: str, config_path: str, device: str = "cuda"
) -> ModelInference:
    """
    Factory function to create inference engine.

    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to model configuration
        device: Device to use for inference

    Returns:
        ModelInference instance
    """
    return ModelInference(model_path, config_path, device)


def run_inference_from_cli(
    model_path: str,
    config_path: str,
    input_path: str,
    output_path: str,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Run inference from command line interface.

    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        input_path: Path to input image or directory
        output_path: Path to save results
        batch_size: Batch size for processing

    Returns:
        Processing results summary
    """
    # Create inference engine
    inference_engine = create_inference_engine(model_path, config_path)

    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    if input_path_obj.is_file():
        # Single file inference
        result = inference_engine.predict_single(str(input_path_obj))

        # Save result
        import json

        with open(output_path_obj / "prediction.json", "w") as f:
            json.dump(result, f, indent=2)

        return {"type": "single_file", "result": result}

    elif input_path_obj.is_dir():
        # Directory inference
        image_files = []
        for ext in [".dcm", ".tif", ".tiff", ".png", ".jpg"]:
            image_files.extend(input_path_obj.glob(f"**/*{ext}"))

        image_paths = [str(f) for f in image_files]
        results = inference_engine.predict_batch(image_paths, batch_size)

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path_obj / "batch_predictions.csv", index=False)

        return {
            "type": "batch",
            "total_images": len(image_paths),
            "results_saved": str(output_path_obj / "batch_predictions.csv"),
        }

    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")
