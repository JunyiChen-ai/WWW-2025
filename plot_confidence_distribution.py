#!/usr/bin/env python3
"""
Plot confidence density distribution for correct vs incorrect predictions
"""
import json
import numpy as np
from pathlib import Path
import argparse

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Plotting libraries not available: {e}")
    print("Please install: pip install matplotlib seaborn scipy")
    PLOTTING_AVAILABLE = False

def load_predictions_with_confidence(file_path):
    """Load predictions with confidence values"""
    print(f"Loading predictions from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def plot_confidence_distribution(predictions_data, output_dir=None, dataset_name="FakeSV"):
    """
    Plot confidence density distribution for correct vs incorrect predictions
    
    Args:
        predictions_data: List of predictions with confidence values
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset
    """
    
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping plot generation.")
        return None
    
    # Extract confidence values and correctness
    correct_confidences = []
    incorrect_confidences = []
    model_type = "unknown"
    
    for pred in predictions_data:
        confidence = pred.get('confidence', None)
        prediction = pred.get('prediction', '')
        ground_truth = pred.get('ground_truth', '')
        
        # Get model type from first entry (should be consistent across all)
        if model_type == "unknown" and 'model_type' in pred:
            model_type = pred['model_type']
        
        if confidence is not None:
            is_correct = (prediction == ground_truth)
            
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
    
    print(f"Correct predictions: {len(correct_confidences)}")
    print(f"Incorrect predictions: {len(incorrect_confidences)}")
    
    if len(correct_confidences) == 0 or len(incorrect_confidences) == 0:
        print("Warning: Not enough data for both categories!")
        return
    
    # Convert to numpy arrays
    correct_confidences = np.array(correct_confidences)
    incorrect_confidences = np.array(incorrect_confidences)
    
    # Calculate statistics
    print(f"\nConfidence Statistics:")
    print(f"Correct predictions - Mean: {correct_confidences.mean():.4f}, Std: {correct_confidences.std():.4f}")
    print(f"Incorrect predictions - Mean: {incorrect_confidences.mean():.4f}, Std: {incorrect_confidences.std():.4f}")
    
    # Perform statistical test
    t_stat, p_value = stats.ttest_ind(correct_confidences, incorrect_confidences)
    print(f"T-test: t-stat={t_stat:.4f}, p-value={p_value:.4e}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Plot density distributions
    sns.kdeplot(correct_confidences, label=f'Correct Predictions (n={len(correct_confidences)})', 
                color='green', linewidth=2.5, alpha=0.8)
    sns.kdeplot(incorrect_confidences, label=f'Incorrect Predictions (n={len(incorrect_confidences)})', 
                color='red', linewidth=2.5, alpha=0.8)
    
    # Add vertical lines for means
    plt.axvline(correct_confidences.mean(), color='green', linestyle='--', alpha=0.7, 
                label=f'Correct Mean: {correct_confidences.mean():.3f}')
    plt.axvline(incorrect_confidences.mean(), color='red', linestyle='--', alpha=0.7,
                label=f'Incorrect Mean: {incorrect_confidences.mean():.3f}')
    
    # Customize plot
    plt.xlabel('Confidence (Max Prediction Probability)', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    title = f'Confidence Distribution: Correct vs Incorrect Predictions\n{dataset_name} Dataset'
    if model_type != "unknown":
        title += f' ({model_type.capitalize()} Classifier)'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Add statistics text box
    stats_text = f'Statistical Test:\nt-statistic: {t_stat:.3f}\np-value: {p_value:.2e}'
    if p_value < 0.001:
        significance = 'Highly Significant (p < 0.001)'
    elif p_value < 0.01:
        significance = 'Very Significant (p < 0.01)'
    elif p_value < 0.05:
        significance = 'Significant (p < 0.05)'
    else:
        significance = 'Not Significant (p ≥ 0.05)'
    
    stats_text += f'\n{significance}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    # Set x-axis limits to [0.5, 1.0] for confidence scores
    plt.xlim(0.5, 1.0)
    
    # Add interpretation text
    interp_text = 'Higher confidence → Better calibrated model\nCorrect predictions should have higher confidence'
    plt.text(0.98, 0.02, interp_text, transform=plt.gca().transAxes, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9, style='italic')
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path("result") / dataset_name / "confidence_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    plot_path_png = output_dir / "confidence_distribution.png"
    plot_path_pdf = output_dir / "confidence_distribution.pdf"
    
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(plot_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\nPlot saved to:")
    print(f"  PNG: {plot_path_png}")
    print(f"  PDF: {plot_path_pdf}")
    
    # Show plot
    plt.show()
    
    return {
        'correct_mean': float(correct_confidences.mean()),
        'correct_std': float(correct_confidences.std()),
        'incorrect_mean': float(incorrect_confidences.mean()),
        'incorrect_std': float(incorrect_confidences.std()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'n_correct': len(correct_confidences),
        'n_incorrect': len(incorrect_confidences)
    }

def plot_confidence_histogram(predictions_data, output_dir=None, dataset_name="FakeSV"):
    """
    Plot confidence histogram for correct vs incorrect predictions
    """
    
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping histogram generation.")
        return None
    
    # Extract confidence values and correctness
    correct_confidences = []
    incorrect_confidences = []
    model_type = "unknown"
    
    for pred in predictions_data:
        confidence = pred.get('confidence', None)
        prediction = pred.get('prediction', '')
        ground_truth = pred.get('ground_truth', '')
        
        # Get model type from first entry (should be consistent across all)
        if model_type == "unknown" and 'model_type' in pred:
            model_type = pred['model_type']
        
        if confidence is not None:
            is_correct = (prediction == ground_truth)
            
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
    
    if len(correct_confidences) == 0 or len(incorrect_confidences) == 0:
        print("Warning: Not enough data for histogram!")
        return
    
    # Create histogram plot
    plt.figure(figsize=(12, 8))
    
    bins = np.linspace(0.5, 1.0, 25)  # Focus on confidence range [0.5, 1.0]
    
    plt.hist(correct_confidences, bins=bins, alpha=0.7, label=f'Correct (n={len(correct_confidences)})', 
             color='green', density=True, edgecolor='black', linewidth=0.5)
    plt.hist(incorrect_confidences, bins=bins, alpha=0.7, label=f'Incorrect (n={len(incorrect_confidences)})', 
             color='red', density=True, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Confidence (Max Prediction Probability)', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Frequency', fontsize=14, fontweight='bold')
    title = f'Confidence Histogram: Correct vs Incorrect Predictions\n{dataset_name} Dataset'
    if model_type != "unknown":
        title += f' ({model_type.capitalize()} Classifier)'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save histogram
    if output_dir is None:
        output_dir = Path("result") / dataset_name / "confidence_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_path = output_dir / "confidence_histogram.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Histogram saved to: {hist_path}")
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot confidence distribution for model predictions')
    parser.add_argument('--prediction_file', type=str, required=True,
                        help='Path to JSON file with predictions and confidence values')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--dataset_name', type=str, default='FakeSV',
                        help='Name of the dataset')
    parser.add_argument('--plot_type', type=str, choices=['density', 'histogram', 'both'], default='both',
                        help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Load predictions
    predictions_data = load_predictions_with_confidence(args.prediction_file)
    
    # Generate plots
    if args.plot_type in ['density', 'both']:
        stats = plot_confidence_distribution(predictions_data, args.output_dir, args.dataset_name)
        if stats:
            print(f"\nStatistics summary: {stats}")
    
    if args.plot_type in ['histogram', 'both']:
        plot_confidence_histogram(predictions_data, args.output_dir, args.dataset_name)

if __name__ == "__main__":
    main()