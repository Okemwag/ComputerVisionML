# Medical Image Analysis - Examples and Notebooks

This directory contains example scripts and Jupyter notebooks demonstrating how to use the medical image analysis system for training, evaluation, and inference.

## 📁 Directory Structure

```
examples/
├── README.md                           # This file
├── train_unet_classification.py        # U-Net training example
├── train_deeplabv3_classification.py   # DeepLabV3+ training example
├── train_multitask.py                  # Multi-task learning example
├── inference_single_image.py           # Single image inference
└── inference_batch_processing.py       # Batch inference processing

notebooks/
├── 01_data_exploration_and_visualization.ipynb    # Data exploration
├── 02_model_comparison_and_analysis.ipynb         # Model comparison
└── 03_inference_examples_and_analysis.ipynb       # Inference examples
```

## 🚀 Getting Started

### Prerequisites

1. Install required dependencies:
```bash
pip install -r ../requirements.txt
```

2. Ensure you have the medical imaging dataset in the `../archive/` directory
3. Have a trained model checkpoint (or use the training scripts to create one)

## 📓 Jupyter Notebooks

### 1. Data Exploration and Visualization
**File:** `notebooks/01_data_exploration_and_visualization.ipynb`

**Purpose:** Comprehensive exploration of the medical imaging dataset

**Features:**
- Dataset overview and metadata analysis
- Image quality and preprocessing analysis
- Data augmentation visualization
- Dataset split analysis
- Performance optimization recommendations

**Usage:**
```bash
cd notebooks
jupyter notebook 01_data_exploration_and_visualization.ipynb
```

### 2. Model Comparison and Analysis
**File:** `notebooks/02_model_comparison_and_analysis.ipynb`

**Purpose:** Compare different model architectures and analyze their performance

**Features:**
- Model architecture comparison (U-Net vs DeepLabV3+)
- Inference speed benchmarking
- Memory usage analysis
- Feature map visualization
- Performance trade-off analysis

**Usage:**
```bash
cd notebooks
jupyter notebook 02_model_comparison_and_analysis.ipynb
```

### 3. Inference Examples and Analysis
**File:** `notebooks/03_inference_examples_and_analysis.ipynb`

**Purpose:** Demonstrate inference on real medical images and analyze results

**Features:**
- Single image inference examples
- Batch processing demonstrations
- Comprehensive results visualization
- Error analysis and interpretability
- Clinical insights and recommendations

**Usage:**
```bash
cd notebooks
jupyter notebook 03_inference_examples_and_analysis.ipynb
```

## 🏋️ Training Scripts

### 1. U-Net Classification Training
**File:** `train_unet_classification.py`

**Purpose:** Train a U-Net model for contrast detection classification

**Usage:**
```bash
python train_unet_classification.py --epochs 50 --batch-size 16 --lr 0.001
```

**Key Features:**
- Optimized for classification tasks
- Efficient training with mixed precision
- Comprehensive logging and visualization
- Automatic checkpoint saving

**Arguments:**
- `--config`: Path to configuration file
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--output-dir`: Output directory for results
- `--resume`: Resume from checkpoint

### 2. DeepLabV3+ Classification Training
**File:** `train_deeplabv3_classification.py`

**Purpose:** Train a DeepLabV3+ model for contrast detection classification

**Usage:**
```bash
python train_deeplabv3_classification.py --epochs 40 --batch-size 12 --lr 0.0005
```

**Key Features:**
- Advanced architecture with atrous convolutions
- ASPP module for multi-scale context
- Transfer learning support
- Memory-efficient training

**Arguments:**
- `--freeze-backbone`: Freeze backbone for transfer learning
- Other arguments similar to U-Net script

### 3. Multi-task Learning Training
**File:** `train_multitask.py`

**Purpose:** Train models for both classification and segmentation tasks

**Usage:**
```bash
python train_multitask.py --epochs 60 --cls-weight 1.0 --seg-weight 2.0
```

**Key Features:**
- Joint training for multiple tasks
- Configurable loss weighting
- Shared feature representations
- Advanced visualization

**Arguments:**
- `--cls-weight`: Weight for classification loss (default: 1.0)
- `--seg-weight`: Weight for segmentation loss (default: 2.0)

## 🔍 Inference Scripts

### 1. Single Image Inference
**File:** `inference_single_image.py`

**Purpose:** Perform inference on a single medical image

**Usage:**
```bash
python inference_single_image.py --model path/to/model.pth --image path/to/image.dcm --save-viz
```

**Features:**
- Detailed single image analysis
- Comprehensive visualization
- Metadata extraction and analysis
- Confidence scoring

**Arguments:**
- `--model`: Path to trained model checkpoint (required)
- `--image`: Path to input image (required)
- `--config`: Path to model configuration
- `--output-dir`: Output directory for results
- `--save-viz`: Save visualization to file

### 2. Batch Processing
**File:** `inference_batch_processing.py`

**Purpose:** Process multiple images in batch for comprehensive analysis

**Usage:**
```bash
python inference_batch_processing.py --model path/to/model.pth --data-dir ../archive --metadata-file ../archive/overview.csv
```

**Features:**
- Efficient batch processing
- Comprehensive performance metrics
- Statistical analysis and visualization
- Detailed results export

**Arguments:**
- `--model`: Path to trained model checkpoint (required)
- `--data-dir`: Path to data directory (required)
- `--metadata-file`: Path to metadata CSV file (required)
- `--batch-size`: Batch size for inference (default: 16)
- `--split`: Dataset split to use (train/val/test)

## 📊 Example Outputs

### Training Outputs
- Model checkpoints (`.pth` files)
- Training curves and metrics plots
- Configuration files
- Comprehensive logs

### Inference Outputs
- Prediction visualizations
- Performance metrics (accuracy, AUC, confusion matrix)
- Detailed results CSV files
- Statistical analysis reports

## 🛠️ Customization

### Configuration Files
All training scripts support YAML configuration files for easy customization:

```yaml
# Example configuration
data:
  data_dir: "../archive"
  metadata_file: "../archive/overview.csv"
  image_size: [256, 256]
  batch_size: 16

model:
  architecture: "unet"
  task_type: "classification"
  num_classes: 2
  depth: 4

training:
  num_epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
```

### Custom Datasets
To use your own dataset:
1. Organize data in the expected directory structure
2. Create a metadata CSV file with required columns
3. Update configuration files accordingly

### Model Architectures
The system supports:
- **U-Net**: Efficient encoder-decoder with skip connections
- **DeepLabV3+**: Advanced architecture with atrous convolutions
- **Multi-task**: Joint classification and segmentation

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient accumulation
   - Use mixed precision training

2. **Data Loading Errors**
   - Check file paths and permissions
   - Verify metadata CSV format
   - Ensure DICOM/TIFF files are valid

3. **Model Loading Issues**
   - Verify checkpoint file integrity
   - Check model configuration compatibility
   - Ensure correct device (CPU/GPU) usage

### Performance Optimization

1. **Training Speed**
   - Use appropriate number of workers
   - Enable pin_memory for GPU training
   - Use mixed precision training

2. **Memory Usage**
   - Adjust batch size based on available memory
   - Use gradient accumulation for larger effective batch sizes
   - Monitor memory usage during training

## 📚 Additional Resources

- [Main README](../README.md) - Project overview and setup
- [API Documentation](../docs/api_reference.md) - Detailed API reference
- [Configuration Guide](../docs/configuration_guide.md) - Configuration options
- [Troubleshooting Guide](../docs/troubleshooting_guide.md) - Common issues and solutions

## 🤝 Contributing

To contribute new examples or improvements:
1. Follow the existing code structure and style
2. Include comprehensive documentation
3. Add appropriate error handling
4. Test with different configurations
5. Update this README if needed

## 📄 License

This project is part of the Medical Image Analysis system. Please refer to the main project license for usage terms.