"""
Evaluation and visualization script for model predictions.

Analyzes model predictions, generates plots, and computes detailed metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import Config


def plot_predictions_vs_actual(y_true, y_pred, save_path=None):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('Actual Dry Weight', fontsize=12)
    plt.ylabel('Predicted Dry Weight', fontsize=12)
    plt.title(f'Predictions vs Actual\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_residuals(y_true, y_pred, save_path=None):
    """Plot residuals distribution."""
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual plot
    axes[0].scatter(y_true, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Actual Dry Weight', fontsize=12)
    axes[0].set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def analyze_results(results_path):
    """
    Analyze prediction results.
    
    Args:
        results_path: Path to CSV with columns [id, actual, predicted]
    """
    print("="*60)
    print("Results Analysis")
    print("="*60)
    
    # Load results
    df = pd.read_csv(results_path)
    print(f"\nLoaded {len(df)} predictions from {results_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract predictions and actuals
    if 'actual' in df.columns and 'predicted' in df.columns:
        y_true = df['actual'].values
        y_pred = df['predicted'].values
    elif 'dry_weight' in df.columns and 'prediction' in df.columns:
        y_true = df['dry_weight'].values
        y_pred = df['prediction'].values
    else:
        print("Error: Could not find actual and predicted columns")
        print("Expected: 'actual' and 'predicted' or 'dry_weight' and 'prediction'")
        return
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print("\n" + "="*60)
    print("Metrics")
    print("="*60)
    print(f"MAE (Mean Absolute Error):     {mae:.6f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"R² Score:                       {r2:.6f}")
    print(f"MAPE (Mean Absolute % Error):   {mape:.4f}%")
    
    # Error analysis
    errors = np.abs(y_pred - y_true)
    print("\n" + "="*60)
    print("Error Analysis")
    print("="*60)
    print(f"Min error:     {errors.min():.6f}")
    print(f"Max error:     {errors.max():.6f}")
    print(f"Median error:  {np.median(errors):.6f}")
    print(f"75th percentile error: {np.percentile(errors, 75):.6f}")
    print(f"95th percentile error: {np.percentile(errors, 95):.6f}")
    
    # Predictions statistics
    print("\n" + "="*60)
    print("Prediction Statistics")
    print("="*60)
    print(f"Actual - Mean: {y_true.mean():.4f}, Std: {y_true.std():.4f}")
    print(f"Predicted - Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}")
    print(f"Actual range: [{y_true.min():.4f}, {y_true.max():.4f}]")
    print(f"Predicted range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # Worst predictions
    worst_idx = np.argsort(errors)[-5:]
    print("\n" + "="*60)
    print("Top 5 Worst Predictions")
    print("="*60)
    for i, idx in enumerate(worst_idx[::-1], 1):
        print(f"{i}. ID: {df.iloc[idx]['id']}, "
              f"Actual: {y_true[idx]:.4f}, "
              f"Predicted: {y_pred[idx]:.4f}, "
              f"Error: {errors[idx]:.4f}")
    
    # Generate plots
    output_dir = Config.OUTPUT_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    plot_predictions_vs_actual(
        y_true, y_pred, 
        save_path=output_dir / "predictions_vs_actual.png"
    )
    
    plot_residuals(
        y_true, y_pred,
        save_path=output_dir / "residuals.png"
    )
    
    print("\n" + "="*60)
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print("="*60)


def main():
    """Main evaluation function."""
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Default: look for validation results
        results_path = Config.OUTPUT_DIR / "validation_results.csv"
        
        if not results_path.exists():
            print("Usage: python evaluate.py <results_csv>")
            print("\nResults CSV should have columns: id, actual, predicted")
            print("Or: id, dry_weight, prediction")
            return
    
    analyze_results(results_path)


if __name__ == '__main__':
    main()
