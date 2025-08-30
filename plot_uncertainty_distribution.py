#!/usr/bin/env python3
"""
Plot uncertainty density distribution for correct vs incorrect predictions
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

def load_predictions_with_uncertainty(file_path):
    """Load predictions with uncertainty values"""
    print(f"Loading predictions from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def plot_uncertainty_distribution(predictions_data, output_dir=None, dataset_name="FakeSV"):
    """
    Plot uncertainty density distribution for correct vs incorrect predictions
    
    Args:
        predictions_data: List of predictions with uncertainty values
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset
    """
    
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping plot generation.")
        return None
    
    # Extract uncertainty values and correctness
    correct_uncertainties = []
    incorrect_uncertainties = []
    
    for pred in predictions_data:
        uncertainty = pred.get('uncertainty', None)
        prediction = pred.get('prediction', '')
        ground_truth = pred.get('ground_truth', '')
        
        if uncertainty is not None:
            is_correct = (prediction == ground_truth)
            
            if is_correct:
                correct_uncertainties.append(uncertainty)
            else:
                incorrect_uncertainties.append(uncertainty)
    
    print(f"Correct predictions: {len(correct_uncertainties)}")
    print(f"Incorrect predictions: {len(incorrect_uncertainties)}")
    
    if len(correct_uncertainties) == 0 or len(incorrect_uncertainties) == 0:
        print("Warning: Not enough data for both categories!")
        return
    
    # Convert to numpy arrays
    correct_uncertainties = np.array(correct_uncertainties)
    incorrect_uncertainties = np.array(incorrect_uncertainties)
    
    # Calculate statistics
    print(f"\nUncertainty Statistics:")
    print(f"Correct predictions - Mean: {correct_uncertainties.mean():.4f}, Std: {correct_uncertainties.std():.4f}")
    print(f"Incorrect predictions - Mean: {incorrect_uncertainties.mean():.4f}, Std: {incorrect_uncertainties.std():.4f}")
    
    # Perform statistical test
    t_stat, p_value = stats.ttest_ind(correct_uncertainties, incorrect_uncertainties)
    print(f"T-test: t-stat={t_stat:.4f}, p-value={p_value:.4e}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Plot density distributions
    sns.kdeplot(correct_uncertainties, label=f'Correct Predictions (n={len(correct_uncertainties)})', 
                color='green', linewidth=2.5, alpha=0.8)
    sns.kdeplot(incorrect_uncertainties, label=f'Incorrect Predictions (n={len(incorrect_uncertainties)})', 
                color='red', linewidth=2.5, alpha=0.8)
    
    # Add vertical lines for means
    plt.axvline(correct_uncertainties.mean(), color='green', linestyle='--', alpha=0.7, 
                label=f'Correct Mean: {correct_uncertainties.mean():.3f}')
    plt.axvline(incorrect_uncertainties.mean(), color='red', linestyle='--', alpha=0.7,
                label=f'Incorrect Mean: {incorrect_uncertainties.mean():.3f}')
    
    # Customize plot
    plt.xlabel('Uncertainty (u)', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.title(f'Uncertainty Distribution: Correct vs Incorrect Predictions\n{dataset_name} Dataset', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Add statistics text box
    stats_text = f'Statistical Test:\nt-statistic: {t_stat:.3f}\np-value: {p_value:.2e}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Set x-axis limits to focus on the data range
    all_uncertainties = np.concatenate([correct_uncertainties, incorrect_uncertainties])
    x_min, x_max = all_uncertainties.min(), all_uncertainties.max()
    x_range = x_max - x_min
    plt.xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path("result") / dataset_name / "uncertainty_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    plot_path_png = output_dir / "uncertainty_distribution.png"
    plot_path_pdf = output_dir / "uncertainty_distribution.pdf"
    
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(plot_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\nPlot saved to:")
    print(f"  PNG: {plot_path_png}")
    print(f"  PDF: {plot_path_pdf}")
    
    # Show plot
    plt.show()
    
    return {
        'correct_mean': float(correct_uncertainties.mean()),
        'correct_std': float(correct_uncertainties.std()),
        'incorrect_mean': float(incorrect_uncertainties.mean()),
        'incorrect_std': float(incorrect_uncertainties.std()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'n_correct': len(correct_uncertainties),
        'n_incorrect': len(incorrect_uncertainties)
    }

def plot_uncertainty_histogram(predictions_data, output_dir=None, dataset_name="FakeSV"):
    """
    Plot uncertainty histogram for correct vs incorrect predictions
    """
    
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping histogram generation.")
        return None
    
    # Extract uncertainty values and correctness
    correct_uncertainties = []
    incorrect_uncertainties = []
    
    for pred in predictions_data:
        uncertainty = pred.get('uncertainty', None)
        prediction = pred.get('prediction', '')
        ground_truth = pred.get('ground_truth', '')
        
        if uncertainty is not None:
            is_correct = (prediction == ground_truth)
            
            if is_correct:
                correct_uncertainties.append(uncertainty)
            else:
                incorrect_uncertainties.append(uncertainty)
    
    if len(correct_uncertainties) == 0 or len(incorrect_uncertainties) == 0:
        print("Warning: Not enough data for histogram!")
        return
    
    # Create histogram plot
    plt.figure(figsize=(12, 8))
    
    bins = np.linspace(0, max(max(correct_uncertainties), max(incorrect_uncertainties)), 30)
    
    plt.hist(correct_uncertainties, bins=bins, alpha=0.7, label=f'Correct (n={len(correct_uncertainties)})', 
             color='green', density=True)
    plt.hist(incorrect_uncertainties, bins=bins, alpha=0.7, label=f'Incorrect (n={len(incorrect_uncertainties)})', 
             color='red', density=True)
    
    plt.xlabel('Uncertainty (u)', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Frequency', fontsize=14, fontweight='bold')
    plt.title(f'Uncertainty Histogram: Correct vs Incorrect Predictions\n{dataset_name} Dataset', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save histogram
    if output_dir is None:
        output_dir = Path("result") / dataset_name / "uncertainty_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_path = output_dir / "uncertainty_histogram.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Histogram saved to: {hist_path}")
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot uncertainty distribution for model predictions')
    parser.add_argument('--prediction_file', type=str, required=True,
                        help='Path to JSON file with predictions and uncertainty values')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--dataset_name', type=str, default='FakeSV',
                        help='Name of the dataset')
    parser.add_argument('--plot_type', type=str, choices=['density', 'histogram', 'both'], default='both',
                        help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Load predictions
    predictions_data = load_predictions_with_uncertainty(args.prediction_file)
    
    # Generate plots
    if args.plot_type in ['density', 'both']:
        stats = plot_uncertainty_distribution(predictions_data, args.output_dir, args.dataset_name)
        print(f"\nStatistics summary: {stats}")
    
    if args.plot_type in ['histogram', 'both']:
        plot_uncertainty_histogram(predictions_data, args.output_dir, args.dataset_name)

if __name__ == "__main__":
    main()