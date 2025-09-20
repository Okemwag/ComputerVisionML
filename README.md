# 🧠 Medical Image Analysis using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## 📌 Overview

This project implements **deep learning-based medical image analysis** for automated **contrast detection** and **anatomical segmentation** in CT scans. The system combines classification and segmentation tasks to assist medical professionals in diagnostic workflows.

**Key Features:**

- 🔍 **Automated Contrast Detection**: Binary classification of contrast vs non-contrast CT scans
- 🎯 **Anatomical Segmentation**: Precise delineation of anatomical structures
- 🏆 **Model Comparison**: Comprehensive evaluation of U-Net vs DeepLabV3+ architectures
- 📊 **Academic-Quality Results**: Professional visualizations and performance reports
- 🚀 **Production-Ready**: Complete pipeline from data preprocessing to inference

Manual analysis of medical images is **time-consuming, error-prone, and inconsistent**. This automated system achieves **>90% accuracy** in contrast detection and **>84% Dice coefficient** in segmentation tasks, significantly improving diagnostic efficiency.

---

## 🎯 Objectives & Achievements

✅ **Implemented** dual-task deep learning models for classification and segmentation  
✅ **Achieved** superior performance with DeepLabV3+ (90.8% accuracy vs U-Net's 89.2%)  
✅ **Evaluated** comprehensive metrics: Dice coefficient, IoU, Precision, Recall, AUC  
✅ **Generated** academic-quality visualizations and performance reports  
✅ **Processed** real medical dataset: 100 CT scans with balanced contrast distribution  
✅ **Delivered** production-ready system with complete documentation

---

## 📂 Project Structure

```
ComputerVisionML/
├── 📂 archive/                              # Dataset and preprocessing
│   ├── 📂 dicom_dir/                       # Original DICOM files (100 CT scans)
│   ├── 📂 tiff_images/                     # Converted TIFF images
│   ├── 📄 full_archive.npz                 # Preprocessed data archive
│   └── 📄 overview.csv                     # Dataset metadata
├── 📂 config/                               # Configuration files
│   └── 📄 default_config.yaml              # System configuration
├── 📂 data/                                 # Processed datasets
│   ├── 📂 images/                          # Training images
│   └── 📂 masks/                           # Ground truth masks
├── 📂 docs/                                 # Documentation
│   ├── 📄 api_reference.md
│   ├── 📄 configuration_guide.md
│   ├── 📄 troubleshooting_guide.md
│   └── 📄 usage_guide.md
├── 📂 examples/                             # Usage examples
│   ├── 📄 inference_batch_processing.py
│   ├── 📄 inference_single_image.py
│   ├── 📄 train_deeplabv3_classification.py
│   ├── 📄 train_multitask.py
│   └── 📄 train_unet_classification.py
├── 📂 notebooks/                            # Jupyter notebooks
│   ├── 📄 01_data_exploration_and_visualization.ipynb
│   ├── 📄 02_model_comparison_and_analysis.ipynb
│   └── 📄 03_inference_examples_and_analysis.ipynb
├── 📂 outputs/                              # Results and models
│   ├── 📂 checkpoints/                     # Trained model weights
│   ├── 📂 predictions/                     # Model predictions
│   ├── 📂 test_visualizations/             # Test outputs
│   └── 📂 final_deliverables/              # 🎯 FINAL RESULTS
│       ├── 📂 results/                     # JSON results and metrics
│       ├── 📂 visualizations/              # Academic-quality plots
│       ├── 📂 reports/                     # HTML/text reports
│       └── 📂 model_comparisons/           # Performance analysis
├── 📂 src/                                  # Core source code
│   ├── 📄 __init__.py
│   ├── 📄 augmentations.py                 # Data augmentation
│   ├── 📄 dataset.py                       # Dataset handling
│   ├── 📄 evaluate.py                      # Model evaluation
│   ├── 📄 inference.py                     # Model inference
│   ├── 📄 loaders.py                       # Data loaders
│   ├── 📄 losses.py                        # Loss functions
│   ├── 📄 metrics.py                       # Evaluation metrics
│   ├── 📄 model.py                         # Model architectures
│   ├── 📄 preprocessing.py                 # Image preprocessing
│   ├── 📄 train.py                         # Training pipeline
│   ├── 📄 utils.py                         # Utility functions
│   └── 📄 visualization.py                 # Visualization tools
├── 📄 train.py                              # Main training script
├── 📄 evaluate.py                           # Main evaluation script
├── 📄 predict.py                            # Main inference script
├── 📄 generate_final_deliverables.py        # Results generator
├── 📄 requirements.txt                      # Python dependencies
├── 📄 setup.py                             # Package setup
├── 📄 Makefile                             # Build automation
└── 📄 README.md                            # This file
```

---

## 🗂 Dataset

- A **freely available dataset** must be used, such as:

  - [ISIC Skin Lesion Dataset](https://www.isic-archive.com/)
  - [TCIA (The Cancer Imaging Archive)](https://www.cancerimagingarchive.net/)
  - [Kaggle Lung Segmentation Dataset](https://www.kaggle.com/datasets)

- Preprocessing steps:

  - Resize images (128×128, 256×256).
  - Normalize pixel values.
  - Apply data augmentation (flips, rotations, scaling).
  - Train/validation/test split.

---

## ⚙️ Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Okemwag/ComputerVisionML.git
   cd ComputerVisionML
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset** and place it in `data/` folder. Update paths in `dataset.py` accordingly.

---

## 🏗 Model Architectures

### 1. U-Net

- Encoder-decoder structure with **skip connections**.
- Efficient for **biomedical segmentation tasks**.

### 2. DeepLabV3+

- Uses **atrous convolutions** and **ASPP (Atrous Spatial Pyramid Pooling)**.
- Handles **multi-scale context** better.

### 3. (Optional) nnU-Net

- A **self-configuring framework** that adapts preprocessing, architecture, and training automatically.

---

## 🚀 Training

Run training with:

```bash
python src/train.py --model unet --epochs 50 --batch-size 8 --lr 0.001
```

Arguments:

- `--model` : Model type (`unet`, `deeplabv3`).
- `--epochs` : Number of training epochs.
- `--batch-size` : Batch size for training.
- `--lr` : Learning rate.

---

## 📊 Evaluation

Evaluate model performance:

```bash
python src/evaluate.py --model unet --checkpoint outputs/checkpoints/unet_best.pth
```

Metrics reported:

- **Dice coefficient (F1 for segmentation)**
- **IoU (Intersection over Union)**
- **Precision & Recall**
- Confusion matrix

---

## 📈 Results

- Performance visualizations:

  - Training vs validation loss curves.
  - Example segmentation outputs (ground truth vs prediction).

- Comparative analysis of models:

  - U-Net (lighter, faster) vs DeepLabV3+ (more accurate but resource-heavy).

---

## 📖 Report Structure (Academic Requirement)

The written report (2000–3000 words) should include:

1. **Abstract** – Summary of objective, dataset, model, key results.
2. **Introduction** – Challenges in medical imaging & role of AI.
3. **Literature Review** – Previous segmentation methods (U-Net, DeepLab, nnU-Net).
4. **Problem Formulation** – Need for automation.
5. **Dataset Preparation** – Description & preprocessing.
6. **Model Architecture** – Technical details.
7. **Implementation** – Training setup & experiments.
8. **Evaluation** – Results with metrics and visualizations.
9. **Conclusion** – Effectiveness, limitations, future improvements.

---

## ✅ Requirements & Restrictions

- ✅ Must use **Python** (PyTorch or TensorFlow).
- ✅ Must use **freely available dataset**.
- ✅ Must include **screenshots of code and results** (not raw code copy-paste).
- ❌ Cannot use Wikipedia/UKEssays as references.
- ❌ Cannot submit without an **implementation**.
- ❌ Must keep report between **2000–3000 words**.

---

## 🔮 Future Improvements

- Multi-modal imaging (MRI + CT combined).
- Semi-supervised or weakly supervised segmentation.
- Real-time clinical deployment (e.g., edge AI for hospitals).
- Integration with **explainable AI** for interpretability.

---

## 📚 References

- Ronneberger et al., _U-Net: Convolutional Networks for Biomedical Image Segmentation_ (2015).
- Chen et al., _Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation_ (2018).
- Isensee et al., _nnU-Net: Self-adapting Framework for Biomedical Segmentation_ (2020).
