#!/usr/bin/env python3
"""
Example Inference Script: Single Image Processing

This script demonstrates how to perform inference on a single medical image
using trained models for contrast detection.

Requirements Addressed:
- 5.1: Visualize model predictions and performance metrics
- 5.2: Generate loss curves and metric plots
- 5.3: Create confusion matrices and ROC curves
- 5.4: Generate overlay visualizations with color-coded regions
"""

import sys
import os
sys.path.append('../')

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging

# Import custom modules
from src.model import create_model
from src.loaders import ImageLoader
from src.preprocessing import MedicalImagePreprocessor
from src.visualization import VisualizationEngine
from src.utils import setup_logging

def load_trained_model(model_path: str, config_path: str = None, device: str = 'cuda'):
    """Load a trained model from checkpoint."""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Try to get config from checkpoint
        config = checkpoint.get('config', {})
        if not config:
            # Default configuration
            config = {
                'model': {
                    'architecture': 'unet',
                    'task_type': 'classification',
                    'in_channels': 1,
                    'num_classes': 2,
                    'depth': 4,
                    'start_filters': 64
                }
            }
    
    # Create model
    model = create_model(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config

def predict_single_image(model, image_path: str, config: dict, device: str = 'cuda'):
    """Perform inference on a single image."""
    
    # Load image
    image_loader = ImageLoader()
    medical_image = image_loader.load_image(image_path)
    
    # Preprocess image
    preprocessor_config = {
        'image_size': config.get('data', {}).get('image_size', [256, 256]),
        'normalize_method': config.get('data', {}).get('normalize_method', 'hounsfield'),
        'window_type': config.get('data', {}).get('window_type', 'soft_tissue')
    }
    
    preprocessor = MedicalImagePreprocessor(preprocessor_config)
    processed_image = preprocessor.preprocess_medical_image(medical_image)
    
    # Convert to tensor and add batch dimension
    image_tensor = processed_image.image_data.unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        
        if isinstance(outputs, dict):
            # Multi-task model
            classification_output = outputs.get('classification', outputs.get('segmentation'))
            segmentation_output = outputs.get('segmentation')
        else:
            # Single task model
            classification_output = outputs
            segmentation_output = None
        
        # Get probabilities
        probabilities = torch.softmax(classification_output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]
    
    results = {
        'original_image': medical_image,
        'processed_image': processed_image,
        'image_tensor': image_tensor,
        'probabilities': probabilities.cpu(),
        'predicted_class': predicted_class.cpu(),
        'confidence': confidence.cpu(),
        'segmentation_output': segmentation_output.cpu() if segmentation_output is not None else None
    }
    
    return results

def visualize_prediction(results: dict, save_path: str = None):
    """Visualize prediction results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    original_img = results['original_image'].image_data.squeeze().numpy()
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Processed image
    processed_img = results['processed_image'].image_data.squeeze().numpy()
    axes[0, 1].imshow(processed_img, cmap='gray')
    axes[0, 1].set_title('Processed Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Prediction visualization
    probabilities = results['probabilities'].squeeze().numpy()
    predicted_class = results['predicted_class'].item()
    confidence = results['confidence'].item()
    
    class_names = ['No Contrast', 'Contrast']
    colors = ['lightcoral', 'lightblue']
    
    bars = axes[0, 2].bar(class_names, probabilities, color=colors, alpha=0.8)
    axes[0, 2].set_title(f'Prediction: {class_names[predicted_class]}\\nConfidence: {confidence:.3f}', 
                        fontsize=14, fontweight='bold')\n    axes[0, 2].set_ylabel('Probability')\n    axes[0, 2].set_ylim(0, 1)\n    axes[0, 2].grid(True, alpha=0.3)\n    \n    # Add value labels on bars\n    for bar, prob in zip(bars, probabilities):\n        height = bar.get_height()\n        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,\n                        f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')\n    \n    # Image histogram\n    axes[1, 0].hist(original_img.flatten(), bins=50, alpha=0.7, color='blue')\n    axes[1, 0].set_title('Original Image Histogram', fontsize=14, fontweight='bold')\n    axes[1, 0].set_xlabel('Pixel Value')\n    axes[1, 0].set_ylabel('Frequency')\n    axes[1, 0].grid(True, alpha=0.3)\n    \n    # Processed image histogram\n    axes[1, 1].hist(processed_img.flatten(), bins=50, alpha=0.7, color='green')\n    axes[1, 1].set_title('Processed Image Histogram', fontsize=14, fontweight='bold')\n    axes[1, 1].set_xlabel('Pixel Value')\n    axes[1, 1].set_ylabel('Frequency')\n    axes[1, 1].grid(True, alpha=0.3)\n    \n    # Segmentation if available\n    if results['segmentation_output'] is not None:\n        seg_output = results['segmentation_output'].squeeze()\n        seg_mask = torch.argmax(seg_output, dim=0).numpy()\n        \n        axes[1, 2].imshow(processed_img, cmap='gray', alpha=0.7)\n        axes[1, 2].imshow(seg_mask, cmap='jet', alpha=0.3)\n        axes[1, 2].set_title('Segmentation Overlay', fontsize=14, fontweight='bold')\n        axes[1, 2].axis('off')\n    else:\n        # Show confidence map or feature visualization\n        axes[1, 2].text(0.5, 0.5, f'Predicted Class: {predicted_class}\\nConfidence: {confidence:.3f}\\n\\nModel Type: Classification Only', \n                        ha='center', va='center', transform=axes[1, 2].transAxes,\n                        fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))\n        axes[1, 2].axis('off')\n    \n    plt.tight_layout()\n    \n    if save_path:\n        plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        print(f\"Visualization saved to: {save_path}\")\n    \n    plt.show()\n\ndef main():\n    parser = argparse.ArgumentParser(description='Single Image Inference')\n    parser.add_argument('--model', type=str, required=True, \n                       help='Path to trained model checkpoint')\n    parser.add_argument('--image', type=str, required=True,\n                       help='Path to input image')\n    parser.add_argument('--config', type=str, \n                       help='Path to model configuration file')\n    parser.add_argument('--output-dir', type=str, default='../outputs/inference',\n                       help='Output directory for results')\n    parser.add_argument('--device', type=str, default='cuda',\n                       help='Device to use for inference')\n    parser.add_argument('--save-viz', action='store_true',\n                       help='Save visualization to file')\n    \n    args = parser.parse_args()\n    \n    # Create output directory\n    output_dir = Path(args.output_dir)\n    output_dir.mkdir(parents=True, exist_ok=True)\n    \n    # Setup logging\n    setup_logging(str(output_dir))\n    logger = logging.getLogger(__name__)\n    \n    # Set device\n    device = args.device if torch.cuda.is_available() else 'cpu'\n    logger.info(f\"Using device: {device}\")\n    \n    try:\n        # Load model\n        logger.info(f\"Loading model from: {args.model}\")\n        model, config = load_trained_model(args.model, args.config, device)\n        logger.info(\"Model loaded successfully\")\n        \n        # Perform inference\n        logger.info(f\"Processing image: {args.image}\")\n        results = predict_single_image(model, args.image, config, device)\n        \n        # Print results\n        predicted_class = results['predicted_class'].item()\n        confidence = results['confidence'].item()\n        probabilities = results['probabilities'].squeeze().numpy()\n        \n        class_names = ['No Contrast', 'Contrast']\n        \n        print(\"\\n\" + \"=\"*50)\n        print(\"INFERENCE RESULTS\")\n        print(\"=\"*50)\n        print(f\"Image: {Path(args.image).name}\")\n        print(f\"Predicted Class: {class_names[predicted_class]}\")\n        print(f\"Confidence: {confidence:.4f}\")\n        print(f\"Probabilities:\")\n        for i, (name, prob) in enumerate(zip(class_names, probabilities)):\n            print(f\"  {name}: {prob:.4f}\")\n        \n        # Image metadata\n        original_image = results['original_image']\n        print(f\"\\nImage Metadata:\")\n        print(f\"  Image ID: {original_image.image_id}\")\n        print(f\"  Format: {original_image.image_format}\")\n        print(f\"  Shape: {original_image.image_data.shape}\")\n        print(f\"  Age: {original_image.age}\")\n        print(f\"  Actual Contrast: {original_image.contrast_label}\")\n        \n        # Check if prediction is correct\n        if original_image.contrast_label is not None:\n            actual_class = int(original_image.contrast_label)\n            is_correct = predicted_class == actual_class\n            print(f\"  Prediction Correct: {is_correct}\")\n        \n        print(\"=\"*50)\n        \n        # Visualize results\n        save_path = None\n        if args.save_viz:\n            image_name = Path(args.image).stem\n            save_path = output_dir / f\"inference_{image_name}.png\"\n        \n        visualize_prediction(results, save_path)\n        \n        # Save detailed results\n        results_file = output_dir / f\"inference_{Path(args.image).stem}_results.txt\"\n        with open(results_file, 'w') as f:\n            f.write(f\"Inference Results for {Path(args.image).name}\\n\")\n            f.write(f\"Model: {args.model}\\n\")\n            f.write(f\"Predicted Class: {class_names[predicted_class]}\\n\")\n            f.write(f\"Confidence: {confidence:.4f}\\n\")\n            f.write(f\"Probabilities:\\n\")\n            for name, prob in zip(class_names, probabilities):\n                f.write(f\"  {name}: {prob:.4f}\\n\")\n            f.write(f\"\\nImage Metadata:\\n\")\n            for key, value in original_image.metadata.items():\n                f.write(f\"  {key}: {value}\\n\")\n        \n        logger.info(f\"Results saved to: {results_file}\")\n        logger.info(\"Inference completed successfully!\")\n        \n    except Exception as e:\n        logger.error(f\"Inference failed with error: {str(e)}\")\n        raise\n\nif __name__ == '__main__':\n    main()