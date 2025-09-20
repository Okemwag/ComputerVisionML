# Configuration Guide

This guide explains all configuration options available in the Medical Image Analysis system.

## Table of Contents

- [Configuration File Structure](#configuration-file-structure)
- [Model Configuration](#model-configuration)
- [Training Configuration](#training-configuration)
- [Data Configuration](#data-configuration)
- [Augmentation Configuration](#augmentation-configuration)
- [Path Configuration](#path-configuration)
- [Logging Configuration](#logging-configuration)
- [Environment Variables](#environment-variables)
- [Configuration Examples](#configuration-examples)

## Configuration File Structure

The system uses YAML configuration files with the following main sections:

```yaml
model:          # Model architecture and parameters
training:       # Training hyperparameters and settings
data:          # Data loading and preprocessing settings
augmentation:  # Data augmentation parameters
paths:         # File and directory paths
logging:       # Logging configuration
```

## Model Configuration

### Basic Model Settings

```yaml
model:
  architecture: "unet"           # Model architecture: "unet" or "deeplabv3"
  in_channels: 1                 # Number of input channels (1 for grayscale CT)
  num_classes: 2                 # Number of output classes
  task_type: "classification"    # Task type: "classification", "segmentation", or "both"
  image_size: [256, 256]        # Input image size [height, width]
  normalize_method: "hounsfield" # Normalization method: "hounsfield", "minmax", "zscore"
```

### U-Net Specific Settings

```yaml
model:
  architecture: "unet"
  depth: 4                      # Network depth (number of down/up blocks)
  start_filters: 64             # Number of filters in first layer
  bilinear: true                # Use bilinear upsampling (vs transpose conv)
  dropout: 0.2                  # Dropout rate for regularization
```

**Parameter Guidelines:**
- `depth`: 3-5 for most medical images (4 is recommended)
- `start_filters`: 32-128 depending on image complexity and GPU memory
- `dropout`: 0.1-0.5 for regularization (0.2 is a good starting point)

### DeepLabV3+ Specific Settings

```yaml
model:
  architecture: "deeplabv3"
  backbone: "custom"            # Backbone architecture
  aspp_dilations: [1, 6, 12, 18] # Atrous convolution dilation rates
```

### Task-Specific Settings

#### Classification Only
```yaml
model:
  task_type: "classification"
  num_classes: 2                # Binary classification (contrast vs no-contrast)
```

#### Segmentation Only
```yaml
model:
  task_type: "segmentation"
  num_classes: 3                # Background + 2 anatomical structures
```

#### Multi-Task (Classification + Segmentation)
```yaml
model:
  task_type: "both"
  num_classes: 2                # For classification head
  segmentation_classes: 3       # For segmentation head
  task_weights:                 # Loss weighting
    classification: 0.7
    segmentation: 0.3
```

## Training Configuration

### Basic Training Settings

```yaml
training:
  num_epochs: 50                # Number of training epochs
  batch_size: 16                # Batch size for training
  learning_rate: 0.001          # Initial learning rate
  weight_decay: 1e-4            # L2 regularization strength
  gradient_accumulation_steps: 1 # Steps to accumulate gradients
  mixed_precision: true         # Enable mixed precision training
  early_stopping_patience: 10   # Epochs to wait before early stopping
```

### Optimizer Settings

```yaml
training:
  optimizer: "adam"             # Optimizer: "adam", "adamw", "sgd"
  
  # Adam/AdamW specific
  betas: [0.9, 0.999]          # Adam beta parameters
  eps: 1e-8                    # Adam epsilon
  
  # SGD specific
  momentum: 0.9                # SGD momentum
  nesterov: true               # Use Nesterov momentum
```

**Optimizer Recommendations:**
- **Adam**: Good default choice, works well for most cases
- **AdamW**: Better weight decay handling, recommended for larger models
- **SGD**: Can achieve better final performance but requires careful tuning

### Learning Rate Scheduling

```yaml
training:
  scheduler: "reduce_on_plateau" # Scheduler type
  
  # ReduceLROnPlateau settings
  scheduler_patience: 5         # Epochs to wait before reducing LR
  scheduler_factor: 0.5         # Factor to reduce LR by
  scheduler_min_lr: 1e-7       # Minimum learning rate
  
  # CosineAnnealingLR settings
  cosine_t_max: 50             # Maximum number of iterations
  cosine_eta_min: 1e-6         # Minimum learning rate
  
  # StepLR settings
  step_size: 20                # Period of learning rate decay
  gamma: 0.1                   # Multiplicative factor of learning rate decay
```

**Scheduler Recommendations:**
- **ReduceLROnPlateau**: Adaptive, good for most cases
- **CosineAnnealingLR**: Smooth decay, good for longer training
- **StepLR**: Simple step decay, good for established training schedules

### Loss Function Configuration

```yaml
training:
  loss_function: "cross_entropy" # Loss function type
  class_weights: "auto"         # Class weighting: "auto", "balanced", or manual weights
  label_smoothing: 0.0          # Label smoothing factor (0.0-0.2)
  
  # Custom loss weights (for multi-task)
  loss_weights:
    classification: 1.0
    segmentation: 1.0
```

## Data Configuration

### Basic Data Settings

```yaml
data:
  data_dir: "archive"                    # Root data directory
  metadata_file: "archive/overview.csv" # Metadata CSV file
  target_size: [256, 256]              # Target image size
  split_ratios: [0.7, 0.15, 0.15]      # Train/validation/test split ratios
  num_workers: 4                        # Number of data loading workers
  pin_memory: true                      # Pin memory for faster GPU transfer
```

### Data Loading Options

```yaml
data:
  use_dicom: true                       # Load DICOM files
  use_tiff: true                        # Load TIFF files
  prefer_dicom: true                    # Prefer DICOM over TIFF when both exist
  cache_images: false                   # Cache loaded images in memory
  validate_files: true                  # Validate file integrity on loading
```

### Preprocessing Settings

```yaml
data:
  normalize_method: "hounsfield"        # Normalization method
  window_type: "soft_tissue"           # Windowing type for CT scans
  
  # Custom windowing parameters
  window_center: 40                     # HU window center
  window_width: 400                     # HU window width
  
  # Intensity clipping
  clip_values: [-1000, 1000]           # Clip intensity values (HU)
  
  # Resampling
  resample_spacing: [1.0, 1.0]         # Target pixel spacing (mm)
  interpolation: "bilinear"             # Interpolation method
```

**Windowing Presets:**
```yaml
# Soft tissue window
window_center: 40
window_width: 400

# Lung window
window_center: -600
window_width: 1600

# Bone window
window_center: 300
window_width: 1500

# Brain window
window_center: 40
window_width: 80
```

## Augmentation Configuration

### Basic Augmentation Settings

```yaml
augmentation:
  enabled: true                         # Enable augmentations
  augmentation_probability: 0.8         # Probability of applying augmentations
  
  # Geometric augmentations
  rotation_limit: 15                    # Rotation angle limit (degrees)
  horizontal_flip: true                 # Enable horizontal flipping
  vertical_flip: false                  # Enable vertical flipping
  scale_limit: 0.1                     # Scale variation limit
  shift_limit: 0.1                     # Translation limit
  
  # Intensity augmentations
  brightness_limit: 0.2                # Brightness variation limit
  contrast_limit: 0.2                  # Contrast variation limit
  gamma_limit: [0.8, 1.2]             # Gamma correction range
  
  # Noise augmentations
  gaussian_noise: 0.01                 # Gaussian noise standard deviation
  gaussian_blur: 0.0                   # Gaussian blur sigma (0 = disabled)
```

### Advanced Augmentation Settings

```yaml
augmentation:
  # Elastic deformation
  elastic_transform: false
  elastic_alpha: 1.0
  elastic_sigma: 50
  
  # Grid distortion
  grid_distortion: false
  grid_num_steps: 5
  grid_distort_limit: 0.3
  
  # Optical distortion
  optical_distortion: false
  optical_distort_limit: 0.05
  optical_shift_limit: 0.05
  
  # Medical-specific augmentations
  simulate_motion: false               # Simulate motion artifacts
  simulate_noise: false                # Simulate acquisition noise
  simulate_bias_field: false           # Simulate bias field inhomogeneity
```

**Augmentation Guidelines:**
- **Rotation**: 10-20 degrees for medical images
- **Flipping**: Horizontal usually OK, vertical depends on anatomy
- **Intensity**: Keep changes subtle (0.1-0.3) to preserve medical characteristics
- **Noise**: Very low levels (0.01-0.05) to avoid unrealistic artifacts

## Path Configuration

### Directory Structure

```yaml
paths:
  # Input paths
  data_dir: "archive"                   # Input data directory
  config_dir: "config"                  # Configuration files directory
  
  # Output paths
  output_dir: "outputs"                 # Main output directory
  checkpoint_dir: "outputs/checkpoints" # Model checkpoints
  visualization_dir: "outputs/visualizations" # Plots and visualizations
  results_dir: "outputs/results"        # Evaluation results
  logs_dir: "outputs/logs"             # Log files
  
  # Temporary paths
  temp_dir: "/tmp/medical_analysis"     # Temporary files
  cache_dir: "cache"                    # Cached data
```

### File Naming Patterns

```yaml
paths:
  # Checkpoint naming
  checkpoint_pattern: "checkpoint_epoch_{epoch:03d}.pth"
  best_model_name: "best_model.pth"
  
  # Log file naming
  log_file_pattern: "training_{timestamp}.log"
  
  # Results naming
  results_pattern: "results_{model}_{timestamp}.json"
  visualization_pattern: "viz_{type}_{timestamp}.png"
```

## Logging Configuration

### Basic Logging Settings

```yaml
logging:
  level: "INFO"                         # Logging level: DEBUG, INFO, WARNING, ERROR
  log_to_file: true                     # Enable file logging
  log_to_console: true                  # Enable console logging
  log_file: "outputs/logs/training.log" # Log file path
  
  # Log formatting
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  
  # Log rotation
  max_log_size: "10MB"                  # Maximum log file size
  backup_count: 5                       # Number of backup log files
```

### Advanced Logging Settings

```yaml
logging:
  # Component-specific logging levels
  loggers:
    "src.train": "DEBUG"
    "src.model": "INFO"
    "src.dataset": "WARNING"
    
  # Metrics logging
  log_metrics_every: 10                 # Log metrics every N batches
  log_gradients: false                  # Log gradient statistics
  log_weights: false                    # Log weight statistics
  
  # Visualization logging
  log_images: true                      # Log sample images
  log_predictions: true                 # Log prediction examples
  log_frequency: 5                      # Log visualizations every N epochs
```

## Environment Variables

### System Configuration

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES="0,1"      # Specify GPU devices
export CUDA_LAUNCH_BLOCKING=1          # Enable CUDA error checking

# Memory settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Data loading
export OMP_NUM_THREADS=4               # OpenMP threads for data loading
export MKL_NUM_THREADS=4               # MKL threads

# Paths
export MEDICAL_DATA_ROOT="/path/to/data"
export MEDICAL_OUTPUT_ROOT="/path/to/outputs"
```

### Performance Tuning

```bash
# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"

# Deterministic behavior
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

## Configuration Examples

### Quick Training Configuration

```yaml
# config/quick_train.yaml
model:
  architecture: "unet"
  in_channels: 1
  num_classes: 2
  task_type: "classification"
  image_size: [128, 128]  # Smaller for faster training

training:
  num_epochs: 20
  batch_size: 32
  learning_rate: 0.01
  early_stopping_patience: 5

data:
  data_dir: "archive"
  metadata_file: "archive/overview.csv"
  num_workers: 2

augmentation:
  enabled: false  # Disable for faster training

paths:
  output_dir: "outputs/quick_test"
```

### High-Quality Training Configuration

```yaml
# config/high_quality.yaml
model:
  architecture: "unet"
  in_channels: 1
  num_classes: 2
  task_type: "classification"
  image_size: [512, 512]  # High resolution
  depth: 5
  start_filters: 64
  dropout: 0.3

training:
  num_epochs: 100
  batch_size: 8  # Smaller batch for high resolution
  learning_rate: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  mixed_precision: true
  gradient_accumulation_steps: 4

data:
  data_dir: "archive"
  metadata_file: "archive/overview.csv"
  target_size: [512, 512]
  num_workers: 8

augmentation:
  enabled: true
  augmentation_probability: 0.9
  rotation_limit: 20
  brightness_limit: 0.3
  contrast_limit: 0.3
  gaussian_noise: 0.02

paths:
  output_dir: "outputs/high_quality"
  checkpoint_dir: "outputs/high_quality/checkpoints"
```

### Multi-Task Configuration

```yaml
# config/multi_task.yaml
model:
  architecture: "unet"
  task_type: "both"
  num_classes: 2  # Classification classes
  segmentation_classes: 4  # Segmentation classes
  task_weights:
    classification: 0.6
    segmentation: 0.4

training:
  num_epochs: 75
  batch_size: 12
  learning_rate: 0.0005
  loss_weights:
    classification: 1.0
    segmentation: 2.0  # Higher weight for segmentation

data:
  data_dir: "archive"
  metadata_file: "archive/overview.csv"
  target_size: [256, 256]

augmentation:
  enabled: true
  # Conservative augmentation for segmentation
  rotation_limit: 10
  horizontal_flip: true
  vertical_flip: false
  brightness_limit: 0.15
  contrast_limit: 0.15
```

### Production Configuration

```yaml
# config/production.yaml
model:
  architecture: "unet"
  in_channels: 1
  num_classes: 2
  task_type: "classification"
  image_size: [256, 256]
  depth: 4
  start_filters: 64
  dropout: 0.2

training:
  num_epochs: 50
  batch_size: 16
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "reduce_on_plateau"
  mixed_precision: true
  early_stopping_patience: 15

data:
  data_dir: "archive"
  metadata_file: "archive/overview.csv"
  split_ratios: [0.8, 0.1, 0.1]  # More training data
  num_workers: 4
  validate_files: true

augmentation:
  enabled: true
  augmentation_probability: 0.7
  rotation_limit: 15
  horizontal_flip: true
  brightness_limit: 0.2
  contrast_limit: 0.2
  gaussian_noise: 0.01

paths:
  output_dir: "outputs/production"
  checkpoint_dir: "outputs/production/checkpoints"
  results_dir: "outputs/production/results"

logging:
  level: "INFO"
  log_to_file: true
  log_file: "outputs/production/logs/training.log"
  log_metrics_every: 50
```

### Configuration Validation

The system automatically validates configurations and provides helpful error messages:

```python
from src.utils import validate_config, load_config

# Load and validate configuration
config = load_config("config/training_config.yaml")

try:
    validate_config(config)
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

Common validation errors and fixes:

1. **Invalid architecture**: Use "unet" or "deeplabv3"
2. **Incompatible image size**: Ensure divisible by 2^depth for U-Net
3. **Invalid split ratios**: Must sum to 1.0
4. **Missing required paths**: Ensure data_dir and metadata_file exist
5. **Invalid optimizer**: Use "adam", "adamw", or "sgd"

This configuration guide covers all available options in the Medical Image Analysis system. For specific use cases, refer to the example configurations and adjust parameters based on your requirements and computational resources.