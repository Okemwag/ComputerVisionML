# Usage Guide

This guide provides step-by-step instructions for using the Medical Image Analysis system.

## Table of Contents

- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training Models](#training-models)
- [Model Evaluation](#model-evaluation)
- [Inference](#inference)
- [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd medical-image-analysis

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Training Example

```python
import yaml
from src.train import create_trainer

# Load configuration
with open('config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create and train model
trainer = create_trainer(config, device="cuda")
history = trainer.fit(num_epochs=10)

print("Training completed!")
print(f"Final validation accuracy: {history['val']['accuracy'][-1]:.2f}%")
```

### 3. Quick Inference

```python
from src.inference import ModelInference

# Load trained model
inference_engine = ModelInference(
    model_path="outputs/checkpoints/best_model.pth",
    config_path="config/default_config.yaml"
)

# Predict on single image
result = inference_engine.predict_single("path/to/ct_scan.dcm")
print(f"Prediction: {'Contrast' if result['prediction'] == 1 else 'No Contrast'}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Data Preparation

### Understanding the Data Structure

The system expects the following directory structure:

```
archive/
├── dicom_dir/           # DICOM files
│   ├── ID_0000_AGE_0060_CONTRAST_1_CT.dcm
│   ├── ID_0001_AGE_0069_CONTRAST_1_CT.dcm
│   └── ...
├── tiff_images/         # TIFF files (optional)
│   ├── ID_0000_AGE_0060_CONTRAST_1_CT.tif
│   ├── ID_0001_AGE_0069_CONTRAST_1_CT.tif
│   └── ...
└── overview.csv         # Metadata file
```

### Metadata File Format

The CSV file should contain the following columns:

```csv
id,Age,Contrast,dicom_name,tiff_name
ID_0000,60,1,ID_0000_AGE_0060_CONTRAST_1_CT.dcm,ID_0000_AGE_0060_CONTRAST_1_CT.tif
ID_0001,69,1,ID_0001_AGE_0069_CONTRAST_1_CT.dcm,ID_0001_AGE_0069_CONTRAST_1_CT.tif
```

### Loading and Exploring Data

```python
from src.dataset import MedicalImageDataset
from src.loaders import ImageLoader

# Create dataset
dataset = MedicalImageDataset(
    data_dir="archive",
    metadata_file="archive/overview.csv",
    target_size=(256, 256),
    split="train"
)

# Get dataset statistics
stats = dataset.get_dataset_statistics()
print(f"Total samples: {stats['total_samples']}")
print(f"Class distribution: {stats['class_distribution']}")
print(f"Age statistics: {stats['age_statistics']}")

# Load a single sample
image, label, age, sample_id = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label.item()}")
print(f"Age: {age.item()}")
```

### Data Validation

```python
from src.loaders import ImageLoader, DataLoadingError

loader = ImageLoader()

# Validate data loading
image_paths = ["archive/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm"]
metadata_list = [{"id": "ID_0000", "Age": 60, "Contrast": 1}]

try:
    medical_images = loader.load_batch(image_paths, metadata_list)
    print(f"Successfully loaded {len(medical_images)} images")
    
    for img in medical_images:
        print(f"Image ID: {img.image_id}")
        print(f"Shape: {img.image_data.shape}")
        print(f"Format: {img.image_format}")
        print(f"Metadata keys: {list(img.metadata.keys())}")
        
except DataLoadingError as e:
    print(f"Data loading failed: {e}")
```

## Training Models

### Basic Training Configuration

Create a configuration file `config/training_config.yaml`:

```yaml
model:
  architecture: "unet"
  in_channels: 1
  num_classes: 2
  depth: 4
  start_filters: 64
  task_type: "classification"
  dropout: 0.2
  image_size: [256, 256]

training:
  num_epochs: 50
  batch_size: 16
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "reduce_on_plateau"
  early_stopping_patience: 10

data:
  data_dir: "archive"
  metadata_file: "archive/overview.csv"
  split_ratios: [0.7, 0.15, 0.15]
  num_workers: 4

paths:
  output_dir: "outputs"
  checkpoint_dir: "outputs/checkpoints"
```

### Training Script

```python
import yaml
import logging
from pathlib import Path
from src.train import create_trainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(config_path: str):
    """Train a medical image analysis model."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(config, device="cuda")
    
    # Start training
    logger.info("Starting training...")
    history = trainer.fit()
    
    # Log final results
    final_train_acc = history['train']['accuracy'][-1]
    final_val_acc = history['val']['accuracy'][-1]
    
    logger.info(f"Training completed!")
    logger.info(f"Final training accuracy: {final_train_acc:.2f}%")
    logger.info(f"Final validation accuracy: {final_val_acc:.2f}%")
    
    return history

if __name__ == "__main__":
    history = train_model("config/training_config.yaml")
```

### Training with Different Architectures

```python
# Train U-Net
unet_config = config.copy()
unet_config['model']['architecture'] = 'unet'
unet_trainer = create_trainer(unet_config)
unet_history = unet_trainer.fit(num_epochs=30)

# Train DeepLabV3+
deeplabv3_config = config.copy()
deeplabv3_config['model']['architecture'] = 'deeplabv3'
deeplabv3_trainer = create_trainer(deeplabv3_config)
deeplabv3_history = deeplabv3_trainer.fit(num_epochs=30)

# Compare results
print(f"U-Net final accuracy: {unet_history['val']['accuracy'][-1]:.2f}%")
print(f"DeepLabV3+ final accuracy: {deeplabv3_history['val']['accuracy'][-1]:.2f}%")
```

### Resume Training from Checkpoint

```python
from src.train import Trainer

# Create trainer
trainer = create_trainer(config)

# Load checkpoint
checkpoint = trainer.load_checkpoint("outputs/checkpoints/checkpoint_epoch_20.pth")
print(f"Resumed from epoch {checkpoint['epoch']}")

# Continue training
remaining_epochs = 50 - checkpoint['epoch']
history = trainer.fit(num_epochs=remaining_epochs)
```

## Model Evaluation

### Basic Evaluation

```python
from src.evaluate import ModelEvaluator
from src.metrics import MedicalMetrics

# Create evaluator
evaluator = ModelEvaluator(
    model_path="outputs/checkpoints/best_model.pth",
    config_path="config/training_config.yaml"
)

# Evaluate on test set
results = evaluator.evaluate_test_set()

print("Test Set Results:")
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1-Score: {results['f1_score']:.3f}")
```

### Detailed Evaluation with Visualizations

```python
from src.visualization import VisualizationEngine

# Create visualization engine
viz_engine = VisualizationEngine(save_dir="outputs/visualizations")

# Generate comprehensive evaluation report
evaluator.generate_evaluation_report(
    output_dir="outputs/evaluation",
    include_visualizations=True,
    include_confusion_matrix=True,
    include_roc_curve=True
)

# Plot training curves
viz_engine.plot_training_curves(history)

# Visualize predictions
test_images, test_labels = evaluator.get_test_samples(num_samples=8)
predictions = evaluator.predict_batch(test_images)
viz_engine.visualize_predictions(test_images, predictions, test_labels)
```

### Cross-Validation

```python
from src.evaluate import cross_validate_model

# Perform 5-fold cross-validation
cv_results = cross_validate_model(
    config=config,
    k_folds=5,
    random_state=42
)

print("Cross-Validation Results:")
print(f"Mean Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
print(f"Mean F1-Score: {cv_results['f1_mean']:.3f} ± {cv_results['f1_std']:.3f}")
```

### Model Comparison

```python
from src.evaluate import compare_models

# Compare different models
model_configs = {
    "UNet_depth3": {"model": {"architecture": "unet", "depth": 3}},
    "UNet_depth4": {"model": {"architecture": "unet", "depth": 4}},
    "DeepLabV3+": {"model": {"architecture": "deeplabv3"}}
}

comparison_results = compare_models(
    base_config=config,
    model_configs=model_configs,
    num_epochs=30
)

# Print comparison table
print("Model Comparison Results:")
print("-" * 60)
print(f"{'Model':<15} {'Accuracy':<10} {'F1-Score':<10} {'Parameters':<12}")
print("-" * 60)
for model_name, results in comparison_results.items():
    print(f"{model_name:<15} {results['accuracy']:<10.3f} {results['f1_score']:<10.3f} {results['parameters']:<12,}")
```

## Inference

### Single Image Prediction

```python
from src.inference import ModelInference

# Initialize inference engine
inference_engine = ModelInference(
    model_path="outputs/checkpoints/best_model.pth",
    config_path="config/training_config.yaml",
    device="cuda"
)

# Predict on single image
result = inference_engine.predict_single("path/to/ct_scan.dcm")

print(f"Image ID: {result['image_id']}")
print(f"Prediction: {'Contrast' if result['prediction'] == 1 else 'No Contrast'}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Processing time: {result['processing_time']:.3f}s")
```

### Batch Processing

```python
from src.inference import BatchProcessor

# Initialize batch processor
batch_processor = BatchProcessor(
    model_path="outputs/checkpoints/best_model.pth",
    config_path="config/training_config.yaml",
    batch_size=16
)

# Process directory of images
results = batch_processor.process_directory(
    input_dir="data/new_scans",
    output_dir="outputs/predictions"
)

print(f"Processed {len(results)} images")
print(f"Contrast detected in {sum(r['prediction'] for r in results)} images")
```

### Export Results

```python
from src.inference import ResultsExporter

# Initialize results exporter
exporter = ResultsExporter(output_dir="outputs/results")

# Export predictions in different formats
exporter.export_to_csv(results, "predictions.csv")
exporter.export_to_json(results, "predictions.json")
exporter.export_visualizations(results, images, "prediction_visualizations")

# Generate summary report
summary = exporter.generate_summary_report(results)
print(summary)
```

## Advanced Usage

### Custom Data Augmentation

```python
from src.augmentations import MedicalAugmentations
import torch

# Create custom augmentation pipeline
custom_augmentations = MedicalAugmentations(
    rotation_limit=20,
    brightness_limit=0.3,
    contrast_limit=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    gaussian_noise=0.02,
    augmentation_probability=0.9
)

# Apply to image
image = torch.randn(1, 256, 256)  # Example image
augmented_image = custom_augmentations(image)
```

### Custom Loss Functions

```python
import torch.nn as nn
from src.losses import LossManager

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        # Add custom loss component
        return ce

# Use custom loss in training
config['training']['custom_loss'] = True
trainer = create_trainer(config)
trainer.criterion = CustomLoss()
```

### Transfer Learning

```python
from src.model import UNet
import torch

# Load pre-trained model
pretrained_model = UNet(in_channels=1, out_channels=2, depth=4)
checkpoint = torch.load("pretrained_model.pth")
pretrained_model.load_state_dict(checkpoint['model_state_dict'])

# Freeze encoder for transfer learning
pretrained_model.freeze_encoder()

# Fine-tune on new dataset
trainer = Trainer(pretrained_model, train_loader, val_loader, config)
history = trainer.fit(num_epochs=20)  # Fewer epochs for fine-tuning
```

### Multi-GPU Training

```python
import torch.nn as nn

# Enable multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

trainer = Trainer(model, train_loader, val_loader, config)
history = trainer.fit()
```

### Hyperparameter Optimization

```python
import optuna
from src.train import create_trainer

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Update config
    config['training']['learning_rate'] = lr
    config['data']['batch_size'] = batch_size
    config['model']['dropout'] = dropout
    
    # Train model
    trainer = create_trainer(config)
    history = trainer.fit(num_epochs=20)
    
    # Return validation accuracy
    return history['val']['accuracy'][-1]

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value:.3f}")
```

### Custom Metrics

```python
from src.metrics import MedicalMetrics
import torch

class CustomMetrics(MedicalMetrics):
    def compute_sensitivity_specificity(self, predictions, targets):
        """Compute sensitivity and specificity."""
        tp = ((predictions == 1) & (targets == 1)).sum().item()
        tn = ((predictions == 0) & (targets == 0)).sum().item()
        fp = ((predictions == 1) & (targets == 0)).sum().item()
        fn = ((predictions == 0) & (targets == 1)).sum().item()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return sensitivity, specificity

# Use custom metrics
custom_metrics = CustomMetrics(num_classes=2, task_type="classification")
sensitivity, specificity = custom_metrics.compute_sensitivity_specificity(predictions, targets)
```

### Model Interpretability

```python
from src.visualization import create_grad_cam, create_attention_maps

# Generate Grad-CAM visualizations
grad_cam_maps = create_grad_cam(
    model=model,
    images=test_images,
    target_layer='encoder.3',  # Specify target layer
    class_idx=1  # Focus on contrast class
)

# Create attention maps
attention_maps = create_attention_maps(
    model=model,
    images=test_images,
    method='integrated_gradients'
)

# Visualize interpretability results
viz_engine.plot_interpretability_results(
    images=test_images,
    grad_cam_maps=grad_cam_maps,
    attention_maps=attention_maps,
    predictions=predictions
)
```

This usage guide covers the most common scenarios and advanced features of the Medical Image Analysis system. For more specific use cases or troubleshooting, refer to the [Troubleshooting Guide](troubleshooting_guide.md) and [API Reference](api_reference.md).