# API Reference

This document provides comprehensive API documentation for the Medical Image Analysis system.

## Table of Contents

- [Core Components](#core-components)
- [Data Loading](#data-loading)
- [Model Architectures](#model-architectures)
- [Training Pipeline](#training-pipeline)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Visualization](#visualization)
- [Utilities](#utilities)

## Core Components

### MedicalImage

A standardized data structure for medical images.

```python
@dataclass
class MedicalImage:
    image_id: str
    image_data: torch.Tensor  # Shape: (C, H, W)
    metadata: Dict[str, Any]
    contrast_label: bool
    age: int
    file_path: str
    image_format: str  # 'dicom' or 'tiff'
```

**Methods:**
- `to_dict() -> Dict[str, Any]`: Convert to dictionary for serialization
- `from_dicom(dicom_path: str, metadata: Dict[str, Any]) -> MedicalImage`: Create from DICOM file

**Example:**
```python
from src.loaders import MedicalImage

# Create from DICOM
medical_image = MedicalImage.from_dicom(
    "path/to/scan.dcm", 
    {"id": "patient_001", "Age": 65, "Contrast": True}
)

# Access image data
image_tensor = medical_image.image_data  # torch.Tensor with shape (1, H, W)
metadata = medical_image.metadata
```

## Data Loading

### ImageLoader

Unified loader for DICOM and TIFF medical images.

```python
class ImageLoader:
    def __init__(self)
    def load_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> MedicalImage
    def load_batch(self, image_paths: list, metadata_list: Optional[list] = None) -> list
```

**Example:**
```python
from src.loaders import ImageLoader

loader = ImageLoader()

# Load single image
medical_image = loader.load_image(
    "data/scans/patient_001.dcm",
    {"id": "001", "Age": 65, "Contrast": True}
)

# Load batch of images
image_paths = ["scan1.dcm", "scan2.dcm"]
metadata_list = [{"id": "001", "Age": 65}, {"id": "002", "Age": 72}]
medical_images = loader.load_batch(image_paths, metadata_list)
```

### DICOMLoader

Specialized loader for DICOM files with metadata extraction.

```python
class DICOMLoader:
    def __init__(self)
    def load_dicom(self, dicom_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]
```

**Extracted Metadata:**
- Patient information (ID, age, sex)
- Study details (date, time, modality)
- Image parameters (slice thickness, pixel spacing)
- Window settings (center, width)
- Contrast information

### TIFFLoader

Loader for TIFF format medical images.

```python
class TIFFLoader:
    def __init__(self)
    def load_tiff(self, tiff_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]
```

### MedicalImageDataset

PyTorch Dataset class for medical images with train/validation/test splitting.

```python
class MedicalImageDataset(torch.utils.data.Dataset):
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
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    )
```

**Methods:**
- `__len__() -> int`: Return dataset length
- `__getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]`: Get sample
- `get_class_weights() -> torch.Tensor`: Calculate class weights for balanced training
- `get_dataset_statistics() -> Dict[str, Any]`: Get comprehensive dataset statistics
- `create_datasets(data_dir: str, metadata_file: str, **kwargs) -> Tuple[Dataset, Dataset, Dataset]`: Create train/val/test splits

**Example:**
```python
from src.dataset import MedicalImageDataset

# Create datasets with automatic splitting
train_dataset, val_dataset, test_dataset = MedicalImageDataset.create_datasets(
    data_dir="archive",
    metadata_file="archive/overview.csv",
    target_size=(256, 256),
    normalize_method="hounsfield"
)

# Get class weights for balanced training
class_weights = train_dataset.get_class_weights()

# Get dataset statistics
stats = train_dataset.get_dataset_statistics()
print(f"Training samples: {stats['current_split_samples']}")
print(f"Class distribution: {stats['class_distribution']}")
```

## Model Architectures

### UNet

U-Net architecture for medical image segmentation and classification.

```python
class UNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 2, 
        depth: int = 4, 
        start_filters: int = 64,
        bilinear: bool = True,
        dropout: float = 0.2,
        task_type: str = 'classification'
    )
```

**Parameters:**
- `in_channels`: Number of input channels (1 for grayscale CT scans)
- `out_channels`: Number of output channels/classes
- `depth`: Depth of the U-Net (number of down/up blocks)
- `start_filters`: Number of filters in first layer
- `bilinear`: Use bilinear upsampling instead of transpose convolution
- `dropout`: Dropout rate for regularization
- `task_type`: 'classification', 'segmentation', or 'both'

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass
- `get_feature_maps(x: torch.Tensor) -> List[torch.Tensor]`: Extract feature maps
- `freeze_encoder()`: Freeze encoder weights for transfer learning
- `unfreeze_encoder()`: Unfreeze encoder weights

**Example:**
```python
from src.model import UNet

# Create U-Net for classification
model = UNet(
    in_channels=1,
    out_channels=2,  # binary classification
    depth=4,
    start_filters=64,
    task_type='classification'
)

# Forward pass
input_tensor = torch.randn(8, 1, 256, 256)  # batch_size=8
output = model(input_tensor)  # Shape: (8, 2)

# Get feature maps for analysis
feature_maps = model.get_feature_maps(input_tensor)
```

### DeepLabV3Plus

DeepLabV3+ architecture with atrous convolutions and ASPP module.

```python
class DeepLabV3Plus(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1, 
        num_classes: int = 2, 
        backbone: str = 'custom',
        task_type: str = 'classification'
    )
```

**Parameters:**
- `in_channels`: Number of input channels
- `num_classes`: Number of output classes
- `backbone`: Backbone architecture ('custom' for simplified version)
- `task_type`: 'classification', 'segmentation', or 'both'

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass
- `extract_features(x: torch.Tensor) -> torch.Tensor`: Extract decoder features
- `freeze_backbone()`: Freeze backbone weights
- `unfreeze_backbone()`: Unfreeze backbone weights

**Example:**
```python
from src.model import DeepLabV3Plus

# Create DeepLabV3+ for classification
model = DeepLabV3Plus(
    in_channels=1,
    num_classes=2,
    task_type='classification'
)

# Forward pass
input_tensor = torch.randn(8, 1, 256, 256)
output = model(input_tensor)  # Shape: (8, 2)
```

### Model Factory Functions

```python
def create_model(config: dict) -> nn.Module
def create_unet_model(config: dict) -> UNet
def create_deeplabv3_model(config: dict) -> DeepLabV3Plus
def count_parameters(model: nn.Module) -> int
def model_summary(model: nn.Module, input_size: tuple = (1, 1, 256, 256)) -> dict
def compare_models(input_size: tuple = (1, 1, 256, 256)) -> dict
```

**Example:**
```python
from src.model import create_model, model_summary, compare_models

# Create model from configuration
config = {
    "model": {
        "architecture": "unet",
        "in_channels": 1,
        "num_classes": 2,
        "depth": 4,
        "start_filters": 64,
        "task_type": "classification"
    }
}
model = create_model(config)

# Get model summary
summary = model_summary(model, input_size=(1, 1, 256, 256))
print(f"Total parameters: {summary['total_parameters']:,}")
print(f"Model size: {summary['model_size_mb']:.2f} MB")

# Compare different architectures
comparison = compare_models()
for name, stats in comparison.items():
    print(f"{name}: {stats['parameters']:,} parameters, {stats['size_mb']:.2f} MB")
```

## Training Pipeline

### Trainer

Main training orchestrator for medical image analysis.

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda"
    )
```

**Methods:**
- `train_epoch() -> Dict[str, float]`: Train for one epoch
- `validate_epoch() -> Dict[str, float]`: Validate for one epoch
- `fit(num_epochs: Optional[int] = None) -> Dict[str, list]`: Train for specified epochs
- `save_checkpoint(epoch: int, metrics: Dict[str, float], is_best: bool = False)`: Save checkpoint
- `load_checkpoint(checkpoint_path: str) -> Dict[str, Any]`: Load checkpoint

**Configuration Options:**
```python
config = {
    "training": {
        "num_epochs": 50,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "optimizer": "adam",  # "adam", "adamw", "sgd"
        "scheduler": "reduce_on_plateau",  # "cosine", "step", None
        "gradient_accumulation_steps": 1,
        "mixed_precision": True,
        "early_stopping_patience": 10
    },
    "model": {
        "architecture": "unet",
        "task_type": "classification",
        "num_classes": 2
    },
    "data": {
        "batch_size": 16,
        "num_workers": 4
    }
}
```

**Example:**
```python
from src.train import create_trainer

# Create trainer from configuration
trainer = create_trainer(config, device="cuda")

# Train the model
history = trainer.fit(num_epochs=50)

# Access training history
train_losses = history['train']['loss']
val_accuracies = history['val']['accuracy']
```

## Evaluation and Metrics

### MedicalMetrics

Medical imaging specific metrics calculator.

```python
class MedicalMetrics:
    def __init__(self, num_classes: int, task_type: str)
    def compute_dice_coefficient(self, predictions: torch.Tensor, targets: torch.Tensor) -> float
    def compute_iou(self, predictions: torch.Tensor, targets: torch.Tensor) -> float
    def compute_classification_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]
    def generate_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray
```

**Example:**
```python
from src.metrics import MedicalMetrics

metrics_calculator = MedicalMetrics(num_classes=2, task_type="classification")

# Compute classification metrics
predictions = torch.tensor([0, 1, 1, 0, 1])
targets = torch.tensor([0, 1, 0, 0, 1])

metrics = metrics_calculator.compute_classification_metrics(predictions, targets)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")

# Generate confusion matrix
cm = metrics_calculator.generate_confusion_matrix(predictions, targets)
```

## Visualization

### VisualizationEngine

Creates plots and visualizations for analysis.

```python
class VisualizationEngine:
    def __init__(self, save_dir: str)
    def plot_training_curves(self, metrics_history: Dict[str, List[float]]) -> None
    def visualize_predictions(self, images: torch.Tensor, predictions: torch.Tensor, 
                            targets: torch.Tensor, num_samples: int = 8) -> None
    def create_segmentation_overlay(self, image: torch.Tensor, mask: torch.Tensor) -> np.ndarray
```

**Example:**
```python
from src.visualization import VisualizationEngine

viz_engine = VisualizationEngine(save_dir="outputs/visualizations")

# Plot training curves
history = {
    'train': {'loss': [0.8, 0.6, 0.4], 'accuracy': [60, 70, 80]},
    'val': {'loss': [0.7, 0.5, 0.45], 'accuracy': [65, 75, 78]}
}
viz_engine.plot_training_curves(history)

# Visualize predictions
viz_engine.visualize_predictions(images, predictions, targets, num_samples=4)
```

## Utilities

### Configuration Management

```python
def load_config(config_path: str) -> Dict[str, Any]
def save_config(config: Dict[str, Any], config_path: str) -> None
def validate_config(config: Dict[str, Any]) -> bool
```

### Data Preprocessing

```python
class MedicalImagePreprocessor:
    def __init__(self, config: Dict[str, Any])
    def preprocess_medical_image(self, medical_image: MedicalImage) -> MedicalImage
    def normalize_hounsfield(self, image: torch.Tensor, window_center: float, window_width: float) -> torch.Tensor
    def resize_image(self, image: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor
```

### Data Augmentation

```python
class MedicalAugmentations:
    def __init__(
        self,
        rotation_limit: int = 15,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        gaussian_noise: float = 0.01,
        augmentation_probability: float = 0.8
    )
    def __call__(self, image: torch.Tensor) -> torch.Tensor
```

## Error Handling

### Custom Exceptions

```python
class MedicalImageAnalysisError(Exception):
    """Base exception for medical image analysis system"""

class DataLoadingError(MedicalImageAnalysisError):
    """Raised when data loading fails"""

class ModelTrainingError(MedicalImageAnalysisError):
    """Raised when training encounters errors"""

class InferenceError(MedicalImageAnalysisError):
    """Raised during inference failures"""

class ValidationError(MedicalImageAnalysisError):
    """Raised when validation fails"""
```

## Configuration Schema

### Complete Configuration Example

```yaml
# config/training_config.yaml
model:
  architecture: "unet"  # "unet" or "deeplabv3"
  in_channels: 1
  num_classes: 2
  depth: 4
  start_filters: 64
  task_type: "classification"  # "classification", "segmentation", "both"
  dropout: 0.2
  image_size: [256, 256]
  normalize_method: "hounsfield"

training:
  num_epochs: 50
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 1e-4
  optimizer: "adam"
  scheduler: "reduce_on_plateau"
  gradient_accumulation_steps: 1
  mixed_precision: true
  early_stopping_patience: 10

data:
  data_dir: "archive"
  metadata_file: "archive/overview.csv"
  target_size: [256, 256]
  normalize_method: "hounsfield"
  window_type: "soft_tissue"
  use_dicom: true
  use_tiff: true
  split_ratios: [0.7, 0.15, 0.15]
  num_workers: 4

augmentation:
  rotation_limit: 15
  brightness_limit: 0.2
  contrast_limit: 0.2
  horizontal_flip: true
  vertical_flip: false
  gaussian_noise: 0.01
  augmentation_probability: 0.8

paths:
  output_dir: "outputs"
  checkpoint_dir: "outputs/checkpoints"
  visualization_dir: "outputs/visualizations"
  results_dir: "outputs/results"

logging:
  level: "INFO"
  log_file: "outputs/training.log"
```