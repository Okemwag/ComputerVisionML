#!/usr/bin/env python3
"""
Test script for visualization functionality.
Tests various visualization components with actual medical data.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.append("src")

from src.dataset import MedicalImageDataset
from src.visualization import VisualizationEngine


def test_training_curves():
    """Test training curve visualization"""
    print("Testing Training Curves Visualization...")

    try:
        viz_engine = VisualizationEngine(save_dir="outputs/test_visualizations")

        # Create dummy training history
        epochs = 20
        metrics_history = {
            "train": {
                "loss": [
                    1.5 - 0.05 * i + 0.1 * np.random.random() for i in range(epochs)
                ],
                "accuracy": [
                    60 + 2 * i + 5 * np.random.random() for i in range(epochs)
                ],
                "f1": [
                    0.6 + 0.02 * i + 0.05 * np.random.random() for i in range(epochs)
                ],
            },
            "val": {
                "loss": [
                    1.4 - 0.04 * i + 0.15 * np.random.random() for i in range(epochs)
                ],
                "accuracy": [
                    62 + 1.8 * i + 8 * np.random.random() for i in range(epochs)
                ],
                "f1": [
                    0.62 + 0.018 * i + 0.08 * np.random.random() for i in range(epochs)
                ],
            },
            "learning_rates": [0.001 * (0.95**i) for i in range(epochs)],
        }

        viz_engine.plot_training_curves(metrics_history, "test_training_curves.png")

        print("✓ Training curves visualization successful")
        print("  Generated: outputs/test_visualizations/test_training_curves.png")

        return True

    except Exception as e:
        print(f"✗ Training curves visualization failed: {e}")
        return False


def test_confusion_matrix():
    """Test confusion matrix visualization"""
    print("\nTesting Confusion Matrix Visualization...")

    try:
        viz_engine = VisualizationEngine(save_dir="outputs/test_visualizations")

        # Create dummy confusion matrix
        confusion_matrix = np.array([[45, 5], [8, 42]])
        class_names = ["No Contrast", "Contrast"]

        viz_engine.plot_confusion_matrix(
            confusion_matrix, class_names, "test_confusion_matrix.png"
        )

        print("✓ Confusion matrix visualization successful")
        print("  Generated: outputs/test_visualizations/test_confusion_matrix.png")

        return True

    except Exception as e:
        print(f"✗ Confusion matrix visualization failed: {e}")
        return False


def test_roc_curves():
    """Test ROC curve visualization"""
    print("\nTesting ROC Curves Visualization...")

    try:
        viz_engine = VisualizationEngine(save_dir="outputs/test_visualizations")

        # Create dummy data for binary classification
        np.random.seed(42)
        n_samples = 100
        targets = np.random.randint(0, 2, n_samples)

        # Create realistic probabilities (better for class 1)
        probabilities = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if targets[i] == 1:
                # Higher probability for correct class
                prob_1 = np.random.beta(3, 1)  # Skewed towards 1
            else:
                # Lower probability for class 1
                prob_1 = np.random.beta(1, 3)  # Skewed towards 0

            probabilities[i] = [1 - prob_1, prob_1]

        class_names = ["No Contrast", "Contrast"]

        viz_engine.plot_roc_curves(
            targets, probabilities, class_names, "test_roc_curves.png"
        )

        print("✓ ROC curves visualization successful")
        print("  Generated: outputs/test_visualizations/test_roc_curves.png")

        return True

    except Exception as e:
        print(f"✗ ROC curves visualization failed: {e}")
        return False


def test_prediction_visualization():
    """Test prediction visualization with real medical images"""
    print("\nTesting Prediction Visualization...")

    try:
        viz_engine = VisualizationEngine(save_dir="outputs/test_visualizations")

        # Load real medical images
        dataset = MedicalImageDataset(
            data_dir="archive",
            metadata_file="archive/overview.csv",
            target_size=(256, 256),
            normalize_method="hounsfield",
            use_dicom=True,
            use_tiff=False,
            split=None,
        )

        # Get a few samples
        images = []
        targets = []
        sample_ids = []

        for i in range(min(8, len(dataset))):
            img, target, age, sample_id = dataset[i]
            images.append(img)
            targets.append(target)
            sample_ids.append(sample_id)

        # Stack images
        images_tensor = torch.stack(images)
        targets_tensor = torch.stack(targets)

        # Create dummy predictions (simulate model output)
        np.random.seed(42)
        predictions_tensor = torch.tensor(
            [np.random.choice([0, 1], p=[0.3, 0.7]) for _ in range(len(images))]
        )

        # Create dummy probabilities
        probabilities_tensor = torch.tensor(
            [[0.3, 0.7] if pred == 1 else [0.8, 0.2] for pred in predictions_tensor]
        ).float()

        viz_engine.visualize_predictions(
            images_tensor,
            predictions_tensor,
            targets_tensor,
            probabilities_tensor,
            num_samples=8,
            save_name="test_predictions.png",
        )

        print("✓ Prediction visualization successful")
        print("  Generated: outputs/test_visualizations/test_predictions.png")
        print(f"  Visualized {len(images)} real CT scan images")

        return True

    except Exception as e:
        print(f"✗ Prediction visualization failed: {e}")
        return False


def test_model_comparison():
    """Test model comparison visualization"""
    print("\nTesting Model Comparison Visualization...")

    try:
        viz_engine = VisualizationEngine(save_dir="outputs/test_visualizations")

        # Create dummy comparison results
        comparison_results = {
            "U-Net": {
                "accuracy": 0.892,
                "precision": 0.885,
                "recall": 0.898,
                "f1_score": 0.891,
                "auc": 0.945,
            },
            "DeepLabV3+": {
                "accuracy": 0.908,
                "precision": 0.902,
                "recall": 0.915,
                "f1_score": 0.908,
                "auc": 0.962,
            },
        }

        viz_engine.plot_model_comparison(
            comparison_results, "test_model_comparison.png"
        )

        print("✓ Model comparison visualization successful")
        print("  Generated: outputs/test_visualizations/test_model_comparison.png")

        return True

    except Exception as e:
        print(f"✗ Model comparison visualization failed: {e}")
        return False


def test_comprehensive_report():
    """Test comprehensive HTML report generation"""
    print("\nTesting Comprehensive Report Generation...")

    try:
        viz_engine = VisualizationEngine(save_dir="outputs/test_visualizations")

        # Create dummy evaluation results
        evaluation_results = {
            "overall_metrics": {
                "accuracy": 0.892,
                "precision": 0.885,
                "recall": 0.898,
                "f1_score": 0.891,
                "auc": 0.945,
            }
        }

        viz_engine.generate_comprehensive_report(evaluation_results, "test_report.html")

        print("✓ Comprehensive report generation successful")
        print("  Generated: outputs/test_visualizations/test_report.html")

        return True

    except Exception as e:
        print(f"✗ Comprehensive report generation failed: {e}")
        return False


def test_segmentation_overlay():
    """Test segmentation overlay creation"""
    print("\nTesting Segmentation Overlay...")

    try:
        viz_engine = VisualizationEngine(save_dir="outputs/test_visualizations")

        # Create dummy image and mask
        image = torch.randn(256, 256)
        mask = torch.zeros(256, 256)

        # Add some "segmented" regions
        mask[100:150, 100:150] = 1  # Square region
        mask[180:220, 180:220] = 1  # Another region

        overlay = viz_engine.create_segmentation_overlay(image, mask, alpha=0.5)

        # Save overlay visualization
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy(), cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask.numpy(), cmap="Reds")
        plt.title("Segmentation Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            viz_engine.save_dir / "test_segmentation_overlay.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("✓ Segmentation overlay successful")
        print("  Generated: outputs/test_visualizations/test_segmentation_overlay.png")

        return True

    except Exception as e:
        print(f"✗ Segmentation overlay failed: {e}")
        return False


def main():
    """Run all visualization tests"""
    print("=" * 60)
    print("Medical Image Visualization Tests")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(test_training_curves())
    results.append(test_confusion_matrix())
    results.append(test_roc_curves())
    results.append(test_prediction_visualization())
    results.append(test_model_comparison())
    results.append(test_comprehensive_report())
    results.append(test_segmentation_overlay())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Training Curves: {'✓ PASS' if results[0] else '✗ FAIL'}")
    print(f"Confusion Matrix: {'✓ PASS' if results[1] else '✗ FAIL'}")
    print(f"ROC Curves: {'✓ PASS' if results[2] else '✗ FAIL'}")
    print(f"Prediction Visualization: {'✓ PASS' if results[3] else '✗ FAIL'}")
    print(f"Model Comparison: {'✓ PASS' if results[4] else '✗ FAIL'}")
    print(f"Comprehensive Report: {'✓ PASS' if results[5] else '✗ FAIL'}")
    print(f"Segmentation Overlay: {'✓ PASS' if results[6] else '✗ FAIL'}")

    if all(results):
        print("\n🎉 All visualization tests passed!")
        print("\nGenerated files in outputs/test_visualizations/:")
        print("  - test_training_curves.png")
        print("  - test_confusion_matrix.png")
        print("  - test_roc_curves.png")
        print("  - test_predictions.png")
        print("  - test_model_comparison.png")
        print("  - test_report.html")
        print("  - test_segmentation_overlay.png")
        return 0
    else:
        print("\n❌ Some visualization tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
