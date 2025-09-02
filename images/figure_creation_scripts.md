# Figure Creation Scripts
## Code Templates for Generating Presentation Figures

---

## 1. Training Performance Visualization

### Training Curves Script
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_training_curves(mlflow_data, save_path):
    """
    Generate training curves comparison for all models
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = ['TruncatedModel', 'SegmentationModel', 'ClassificationModel', 
              'MultiModalMultiScale', 'MultiModalMultiFusion']
    
    # Training Loss
    axes[0,0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    for model in models:
        axes[0,0].plot(mlflow_data[model]['train_loss'], label=model, linewidth=2)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[0,1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    for model in models:
        axes[0,1].plot(mlflow_data[model]['val_loss'], label=model, linewidth=2)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Training Accuracy
    axes[1,0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    for model in models:
        axes[1,0].plot(mlflow_data[model]['train_acc'], label=model, linewidth=2)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Validation Accuracy
    axes[1,1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    for model in models:
        axes[1,1].plot(mlflow_data[model]['val_acc'], label=model, linewidth=2)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_curves_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# Usage
# plot_training_curves(mlflow_experiment_data, "/home/youven/Downloads/presentation_carbure/figures/training_metrics")
```

---

## 2. Confusion Matrix Grid

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrices_grid(predictions_dict, true_labels, class_names, save_path):
    """
    Create grid of confusion matrices for all models
    """
    models = list(predictions_dict.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, model_name in enumerate(models):
        if i < len(axes):
            cm = confusion_matrix(true_labels, predictions_dict[model_name])
            
            # Calculate accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[i])
            
            axes[i].set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', 
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
    
    # Hide unused subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrices_grid.png", dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 3. XAI Attribution Visualization

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_attribution_overlays(images, attributions, class_labels, save_path):
    """
    Create attribution overlay visualizations
    """
    n_samples = len(images)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Original Image\nClass: {class_labels[i]}', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Attribution heatmap
        attribution_map = attributions[i]
        im = axes[i, 1].imshow(attribution_map, cmap='RdYlBu_r', alpha=0.8)
        axes[i, 1].set_title('Attribution Heatmap', fontweight='bold')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay = images[i].copy()
        attribution_normalized = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min())
        
        # Create colored overlay
        colored_attribution = plt.cm.RdYlBu_r(attribution_normalized)[:,:,:3]
        overlay_combined = 0.6 * overlay + 0.4 * colored_attribution
        
        axes[i, 2].imshow(overlay_combined)
        axes[i, 2].set_title('Attribution Overlay', fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/attribution_overlays.png", dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 4. Integrated Gradients Profiles

```python
def plot_integrated_gradients_profiles(attributions_per_class, class_names, save_path):
    """
    Plot integrated gradients profiles for each class
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        if i < len(axes):
            # Get attribution data for this class
            class_attributions = attributions_per_class[class_name]
            
            # Calculate statistics
            mean_attribution = np.mean(class_attributions, axis=0)
            std_attribution = np.std(class_attributions, axis=0)
            
            # Plot mean with error bands
            x_coords = np.arange(len(mean_attribution))
            axes[i].plot(x_coords, mean_attribution, linewidth=2, label='Mean Attribution')
            axes[i].fill_between(x_coords, 
                               mean_attribution - std_attribution,
                               mean_attribution + std_attribution,
                               alpha=0.3, label='±1 Std')
            
            axes[i].set_title(f'Attribution Profile: {class_name}', fontweight='bold')
            axes[i].set_xlabel('Spatial Region')
            axes[i].set_ylabel('Attribution Magnitude')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(class_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/integrated_gradients_profiles.png", dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 5. Clustering Visualization

```python
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_decision_clustering(features, labels, predictions, save_path):
    """
    Create t-SNE clustering visualization of model decisions
    """
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Create clustering visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot by true labels
    scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=labels, cmap='tab10', alpha=0.7, s=50)
    axes[0].set_title('t-SNE Visualization: True Labels', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0])
    
    # Plot by predictions
    scatter = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=predictions, cmap='tab10', alpha=0.7, s=50)
    axes[1].set_title('t-SNE Visualization: Predictions', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/decision_clustering.png", dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 6. Performance Comparison Radar Chart

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_performance_radar(model_metrics, model_names, save_path):
    """
    Create radar chart comparing model performance across multiple metrics
    """
    # Define metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed', 'Memory Efficiency']
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot data for each model
    for i, model_name in enumerate(model_names):
        values = model_metrics[model_name]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Model Performance Comparison', size=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 7. MLflow Experiment Comparison

```python
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mlflow_comparison(experiment_id, save_path):
    """
    Create MLflow experiment comparison visualization
    """
    # Get experiment data
    experiment = mlflow.get_experiment(experiment_id)
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy vs Training Time
    axes[0, 0].scatter(runs['training_time'], runs['accuracy'], alpha=0.7, s=100)
    axes[0, 0].set_xlabel('Training Time (minutes)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Training Time', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Parameter importance
    important_params = ['learning_rate', 'batch_size', 'hidden_dim']
    for i, param in enumerate(important_params):
        if param in runs.columns:
            axes[0, 1].scatter(runs[param], runs['accuracy'], 
                              label=param, alpha=0.7, s=100)
    axes[0, 1].set_xlabel('Parameter Value')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Parameter vs Accuracy', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training progression
    best_run = runs.loc[runs['accuracy'].idxmax()]
    axes[1, 0].plot(best_run['epoch'], best_run['train_loss'], label='Train Loss')
    axes[1, 0].plot(best_run['epoch'], best_run['val_loss'], label='Val Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Best Run Training Progression', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Model comparison
    model_performance = runs.groupby('model_type')['accuracy'].agg(['mean', 'std'])
    model_names = model_performance.index
    means = model_performance['mean']
    stds = model_performance['std']
    
    axes[1, 1].bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
    axes[1, 1].set_xlabel('Model Type')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Model Performance Comparison', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/mlflow_experiment_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 8. Data Distribution Analysis

```python
def plot_data_distribution(dataset_info, save_path):
    """
    Create comprehensive data distribution analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Class distribution
    class_counts = dataset_info['class_distribution']
    axes[0, 0].pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    axes[0, 0].set_title('Class Distribution', fontweight='bold')
    
    # Image resolution distribution
    resolutions = dataset_info['image_resolutions']
    axes[0, 1].hist(resolutions, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Image Resolution (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Image Resolution Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bounding box size distribution
    bbox_areas = dataset_info['bbox_areas']
    axes[1, 0].hist(bbox_areas, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Bounding Box Area (pixels²)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Bounding Box Size Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Aspect ratio distribution
    aspect_ratios = dataset_info['aspect_ratios']
    axes[1, 1].hist(aspect_ratios, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Aspect Ratio (width/height)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Aspect Ratio Distribution', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/data_distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
```

---

## Master Script for Figure Generation

```python
def generate_all_figures():
    """
    Master function to generate all presentation figures
    """
    base_path = "/home/youven/Downloads/presentation_carbure/figures"
    
    # Create directories
    import os
    os.makedirs(f"{base_path}/training_metrics", exist_ok=True)
    os.makedirs(f"{base_path}/performance_evaluation", exist_ok=True)
    os.makedirs(f"{base_path}/xai_visualizations", exist_ok=True)
    os.makedirs(f"{base_path}/clustering_analysis", exist_ok=True)
    os.makedirs(f"{base_path}/experiment_tracking", exist_ok=True)
    os.makedirs(f"{base_path}/data_analysis", exist_ok=True)
    
    print("Figure generation directories created.")
    print("Load your data and call individual plotting functions.")
    print("Example usage:")
    print("plot_training_curves(mlflow_data, f'{base_path}/training_metrics')")
    print("plot_confusion_matrices_grid(predictions, labels, classes, f'{base_path}/performance_evaluation')")

# Run master script
if __name__ == "__main__":
    generate_all_figures()
```
