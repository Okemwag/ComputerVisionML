"""
Medical Image Analysis - Evaluation Pipeline

Comprehensive model evaluation with medical imaging metrics and visualizations.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .model import create_model
from .dataset import MedicalImageDataset
from .metrics import MedicalMetrics, CheckpointManager
from .losses import compute_dice_coefficient, compute_iou


class ModelEvaluator:
    """Comprehensive model evaluation pipeline."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize metrics calculator
        model_config = config.get('model', {})
        self.medical_metrics = MedicalMetrics(
            num_classes=model_config.get('num_classes', 2),
            task_type=model_config.get('task_type', 'classification')
        )
        
        # Results storage
        self.evaluation_results = {}
        self.predictions_cache = {}
    
    def evaluate_dataset(self, dataloader: torch.utils.data.DataLoader, dataset_name: str = 'test') -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        self.logger.info(f"Evaluating on {dataset_name} dataset...")
        
        with torch.no_grad():
            for batch_idx, (images, labels, ages, sample_ids) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Handle different output types
                if isinstance(outputs, dict):
                    # Multi-task model
                    cls_outputs = outputs['classification']
                    probabilities = F.softmax(cls_outputs, dim=1)
                    predictions = torch.argmax(cls_outputs, dim=1)
                else:
                    # Single task model
                    probabilities = F.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self.medical_metrics.compute_classification_metrics(
            torch.from_numpy(all_predictions), 
            torch.from_numpy(all_targets)
        )
        
        # Store results
        results = {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'confusion_matrix': self.medical_metrics.generate_confusion_matrix(
                torch.from_numpy(all_predictions),
                torch.from_numpy(all_targets)
            )
        }
        
        self.evaluation_results[dataset_name] = results
        self.predictions_cache[dataset_name] = {
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        # Log results
        self.logger.info(f"{dataset_name.capitalize()} Results:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        
        return results
    
    def cross_validate(self, dataset: MedicalImageDataset, k_folds: int = 5) -> Dict[str, Any]:
        """Perform k-fold cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        
        self.logger.info(f"Performing {k_folds}-fold cross-validation...")
        
        # Get all data
        all_samples = []
        all_labels = []
        
        for i in range(len(dataset)):
            _, label, _, _ = dataset[i]
            all_samples.append(i)
            all_labels.append(label.item())
        
        # Create folds
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, all_labels)):
            self.logger.info(f"Evaluating fold {fold + 1}/{k_folds}")
            
            # Create subset datasets
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=16, shuffle=False, num_workers=2
            )
            
            # Evaluate on this fold
            fold_result = self.evaluate_dataset(val_loader, f'fold_{fold}')
            fold_results.append(fold_result['metrics'])
        
        # Aggregate results
        cv_results = {}
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        self.logger.info("Cross-validation results:")
        for metric, value in cv_results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return {
            'cv_metrics': cv_results,
            'fold_results': fold_results
        }
    
    def compare_models(self, model_configs: List[Dict[str, Any]], test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Compare multiple model configurations."""
        comparison_results = {}
        
        for i, config in enumerate(model_configs):
            model_name = config.get('name', f'model_{i}')
            self.logger.info(f"Evaluating {model_name}...")
            
            # Create and load model
            model = create_model(config)
            
            # Load checkpoint if specified
            if 'checkpoint_path' in config:
                checkpoint_manager = CheckpointManager()
                checkpoint_manager.load_checkpoint(
                    config['checkpoint_path'], model, device=self.device
                )
            
            # Create evaluator for this model
            evaluator = ModelEvaluator(model, config, self.device)
            
            # Evaluate
            results = evaluator.evaluate_dataset(test_loader, 'test')
            comparison_results[model_name] = results['metrics']
        
        return comparison_results
    
    def generate_detailed_report(self, dataset_name: str = 'test') -> Dict[str, Any]:
        """Generate detailed evaluation report."""
        if dataset_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {dataset_name}")
        
        results = self.evaluation_results[dataset_name]
        
        # Classification report
        class_names = [f'Class_{i}' for i in range(self.medical_metrics.num_classes)]
        if self.medical_metrics.num_classes == 2:
            class_names = ['No Contrast', 'Contrast']
        
        cls_report = classification_report(
            results['targets'], 
            results['predictions'],
            target_names=class_names,
            output_dict=True
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_mask = results['targets'] == i
            if class_mask.sum() > 0:
                class_preds = results['predictions'][class_mask]
                class_targets = results['targets'][class_mask]
                
                per_class_metrics[class_name] = {
                    'count': int(class_mask.sum()),
                    'accuracy': float((class_preds == class_targets).mean()),
                    'precision': cls_report[class_name]['precision'],
                    'recall': cls_report[class_name]['recall'],
                    'f1_score': cls_report[class_name]['f1-score']
                }
        
        detailed_report = {
            'overall_metrics': results['metrics'],
            'classification_report': cls_report,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'class_names': class_names,
            'total_samples': len(results['targets'])
        }
        
        return detailed_report
    
    def save_results(self, output_dir: str = 'outputs/evaluation'):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed reports
        for dataset_name in self.evaluation_results.keys():
            report = self.generate_detailed_report(dataset_name)
            
            # Save as JSON
            import json
            with open(output_path / f'{dataset_name}_evaluation_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'predictions': self.predictions_cache[dataset_name]['predictions'],
                'targets': self.predictions_cache[dataset_name]['targets'],
                'probabilities_class_0': self.predictions_cache[dataset_name]['probabilities'][:, 0],
                'probabilities_class_1': self.predictions_cache[dataset_name]['probabilities'][:, 1] if self.predictions_cache[dataset_name]['probabilities'].shape[1] > 1 else 0
            })
            predictions_df.to_csv(output_path / f'{dataset_name}_predictions.csv', index=False)
        
        self.logger.info(f"Evaluation results saved to {output_path}")


def create_evaluator(config: Dict[str, Any], checkpoint_path: Optional[str] = None, device: str = 'cuda') -> ModelEvaluator:
    """
    Factory function to create evaluator from configuration.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Optional path to model checkpoint
        device: Device to use for evaluation
        
    Returns:
        ModelEvaluator instance
    """
    # Create model
    model = create_model(config)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.load_checkpoint(checkpoint_path, model, device=device)
    
    return ModelEvaluator(model, config, device)


def evaluate_model_from_config(config_path: str, checkpoint_path: str, data_dir: str, metadata_file: str) -> Dict[str, Any]:
    """
    Evaluate model from configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing test data
        metadata_file: Path to metadata CSV file
        
    Returns:
        Evaluation results dictionary
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = create_evaluator(config, checkpoint_path)
    
    # Create test dataset
    test_dataset = MedicalImageDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        split='test',
        target_size=tuple(config.get('model', {}).get('image_size', [256, 256])),
        normalize_method=config.get('model', {}).get('normalize_method', 'hounsfield')
    )
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=2
    )
    
    # Evaluate
    results = evaluator.evaluate_dataset(test_loader, 'test')
    
    # Save results
    evaluator.save_results()
    
    return results