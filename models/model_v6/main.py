"""
Main entry point for Model_v6.
Provides command-line interface for setup, preprocessing, training, prediction, and visualization.
"""

import argparse
import sys
from pathlib import Path

from config import Config
from setup import setup_model_v6
from preprocess_dataset import main as preprocess_main
from train import train_model
from dataloader import create_dataloader
from predict import predict_on_test_set
from visualize import create_comprehensive_visualization


def setup(args):
    """Run setup command."""
    version = args.version or "6.1"
    print(f"Setting up Model_v6 version {version}...")
    setup_model_v6(version)


def preprocess(args):
    """Run preprocessing command."""
    print("Running preprocessing...")
    preprocess_main()


def train(args):
        csv_file=config.AUGMENTED_CSV,
        rgb_dir=config.AUGMENTED_RGB_DIR,
        depth_dir=config.AUGMENTED_DEPTH_DIR,
    
    print("Creating dataloaders...")
        num_workers=config.NUM_WORKERS,
        include_target=True
        rgb_dir=f"{config.AUGMENTED_OUTPUT_DIR}/RGBImages",
        rgbd_dir=f"{config.AUGMENTED_OUTPUT_DIR}/RGBDImages",
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        depth_dir=config.TEST_DEPTH_DIR,
    )
    
        num_workers=config.NUM_WORKERS,
        include_target=True


def predict(args):
    """Run prediction command."""
    config = Config()
    
    model_path = args.model or str(Path(config.EXPERIMENT_DIR) / "best_model.pth")
    output_csv = args.output or str(Path(config.EXPERIMENT_DIR) / "predictions.csv")
    
    print(f"Loading model from {model_path}...")
    predictions_df = predict_on_test_set(model_path, config, output_csv)
    
    print(f"\nPrediction Summary:")
    print(f"Mean Absolute Error: {predictions_df['abs_error'].mean():.4f}")
    print(f"RMSE: {(predictions_df['error'] ** 2).mean() ** 0.5:.4f}")
    print(f"\nPredictions saved to {output_csv}")


def visualize(args):
    """Run visualization command."""
    config = Config()
    
    model_path = args.model or str(Path(config.EXPERIMENT_DIR) / "best_model.pth")
    predictions_csv = args.predictions or str(Path(config.EXPERIMENT_DIR) / "predictions.csv")
    output_dir = args.output or str(Path(config.EXPERIMENT_DIR) / "visualizations")
    
    print(f"Creating visualizations...")
    create_comprehensive_visualization(model_path, predictions_csv, config, output_dir)
    print(f"Visualizations saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Model_v6: Dual-Branch CNN for Plant Dry Weight Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup --version 6.1
  python main.py preprocess
  python main.py train
  python main.py predict --model experiments/6.1/best_model.pth
  python main.py visualize --model experiments/6.1/best_model.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup experiment directory')
    setup_parser.add_argument('--version', type=str, help='Version number (e.g., 6.1)')
    setup_parser.set_defaults(func=setup)
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess dataset')
    preprocess_parser.set_defaults(func=preprocess)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.set_defaults(func=train)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    predict_parser.add_argument('--output', type=str, help='Output CSV path')
    predict_parser.set_defaults(func=predict)
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    visualize_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    visualize_parser.add_argument('--predictions', type=str, help='Path to predictions CSV')
    visualize_parser.add_argument('--output', type=str, help='Output directory for visualizations')
    visualize_parser.set_defaults(func=visualize)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
