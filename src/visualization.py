"""
Medical Image Analysis - Visualization and Reporting

Visualization utilities for training curves, predictions, and medical imaging results.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd


class VisualizationEngine:
    """Comprehensive visualization engine for medical image analysis."""
    
    def __init__(self, save_dir: str = 'outputs/visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, metrics_history: Dict[str, List[float]], save_name: str = 'training_curves.png'):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Loss curves
        if 'train' in metrics_history and 'val' in metrics_history:
            epochs = range(1, len(metrics_history['train']['loss']) + 1)
            
            axes[0, 0].plot(epochs, metrics_history['train']['loss'], 'b-', label='Training Loss')
            axes[0, 0].plot(epochs, metrics_history['val']['loss'], 'r-', label='Validation Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy curves
            axes[0, 1].plot(epochs, metrics_history['train']['accuracy'], 'b-', label='Training Accuracy')
            axes[0, 1].plot(epochs, metrics_history['val']['accuracy'], 'r-', label='Validation Accuracy')
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # F1 Score curves
            if 'f1' in metrics_history['train']:
                axes[1, 0].plot(epochs, metrics_history['train']['f1'], 'b-', label='Training F1')
                axes[1, 0].plot(epochs, metrics_history['val']['f1'], 'r-', label='Validation F1')
                axes[1, 0].set_title('F1 Score Curves')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate curve (if available)
            if 'learning_rates' in metrics_history:
                axes[1, 1].plot(epochs, metrics_history['learning_rates'], 'g-')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved to {save_path}")
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, class_names: List[str], save_name: str = 'confusion_matrix.png'):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curves(self, targets: np.ndarray, probabilities: np.ndarray, class_names: List[str], save_name: str = 'roc_curves.png'):
        """Plot ROC curves for classification."""
        plt.figure(figsize=(10, 8))
        
        if probabilities.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
        else:  # Multi-class
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            # Binarize targets
            targets_bin = label_binarize(targets, classes=range(len(class_names)))
            
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(targets_bin[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"ROC curves saved to {save_path}")
    
    def plot_precision_recall_curves(self, targets: np.ndarray, probabilities: np.ndarray, class_names: List[str], save_name: str = 'pr_curves.png'):
        """Plot Precision-Recall curves."""
        plt.figure(figsize=(10, 8))
        
        if probabilities.shape[1] == 2:  # Binary classification
            precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.2f})')
            
        else:  # Multi-class
            from sklearn.preprocessing import label_binarize
            
            targets_bin = label_binarize(targets, classes=range(len(class_names)))
            
            for i in range(len(class_names)):
                precision, recall, _ = precision_recall_curve(targets_bin[:, i], probabilities[:, i])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2,
                        label=f'{class_names[i]} (AUC = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Precision-Recall curves saved to {save_path}")
    
    def visualize_predictions(self, images: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor, probabilities: torch.Tensor = None, num_samples: int = 8, save_name: str = 'predictions.png'):
        """Visualize model predictions on sample images."""
        num_samples = min(num_samples, images.size(0))
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(num_samples):
            # Get image
            img = images[i].squeeze().cpu().numpy()
            
            # Get prediction info
            pred = predictions[i].item() if hasattr(predictions[i], 'item') else predictions[i]
            target = targets[i].item() if hasattr(targets[i], 'item') else targets[i]
            
            # Get confidence if available
            confidence = ""
            if probabilities is not None:
                conf = probabilities[i].max().item() if hasattr(probabilities[i], 'max') else probabilities[i].max()
                confidence = f" (conf: {conf:.2f})"
            
            # Plot image
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Pred: {pred}, True: {target}{confidence}')
            axes[i].axis('off')
            
            # Color border based on correctness
            if pred == target:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)
            else:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Prediction visualizations saved to {save_path}")
    
    def create_segmentation_overlay(self, image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
        """Create segmentation overlay visualization."""
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        
        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Create RGB image
        rgb_image = np.stack([image, image, image], axis=-1)
        
        # Create colored mask
        colored_mask = np.zeros_like(rgb_image)
        colored_mask[mask > 0] = [1, 0, 0]  # Red for segmented regions
        
        # Blend image and mask
        overlay = rgb_image * (1 - alpha) + colored_mask * alpha
        
        return overlay
    
    def plot_model_comparison(self, comparison_results: Dict[str, Dict[str, float]], save_name: str = 'model_comparison.png'):
        """Plot model comparison results."""
        # Extract metrics
        models = list(comparison_results.keys())
        metrics = list(comparison_results[models[0]].keys())
        
        # Create subplot for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [comparison_results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(max(models, key=len)) > 8:
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model comparison saved to {save_path}")
    
    def generate_comprehensive_report(self, evaluation_results: Dict[str, Any], save_name: str = 'evaluation_report.html'):
        """Generate comprehensive HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical Image Analysis - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 30px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                .image-container img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Medical Image Analysis Evaluation Report</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Overall Performance Metrics</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        # Add metrics to table
        if 'overall_metrics' in evaluation_results:
            for metric, value in evaluation_results['overall_metrics'].items():
                html_content += f"""
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{value:.4f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="image-container">
                    <h3>Training Curves</h3>
                    <img src="training_curves.png" alt="Training Curves">
                </div>
                <div class="image-container">
                    <h3>Confusion Matrix</h3>
                    <img src="confusion_matrix.png" alt="Confusion Matrix">
                </div>
                <div class="image-container">
                    <h3>ROC Curves</h3>
                    <img src="roc_curves.png" alt="ROC Curves">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        save_path = self.save_dir / save_name
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Comprehensive report saved to {save_path}")


def create_visualization_engine(config: Dict[str, Any]) -> VisualizationEngine:
    """Factory function to create visualization engine from configuration."""
    paths_config = config.get('paths', {})
    viz_dir = paths_config.get('visualizations_dir', 'outputs/visualizations')
    
    return VisualizationEngine(save_dir=viz_dir)