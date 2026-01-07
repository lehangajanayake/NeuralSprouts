#!/usr/bin/env python
"""
Command-line interface for the multimodal fusion model.

Provides a unified interface for all operations:
- verify: Check data structure
- test: Test model architecture
- train: Run k-fold training
- predict: Generate predictions
- evaluate: Analyze results
"""

import sys
import argparse


def verify_data():
    """Run data verification."""
    from verify_data import main
    main()


def test_model():
    """Test model architecture."""
    from test_model import main
    main()


def train_model():
    """Train the model."""
    from train import main
    main()


def predict():
    """Generate predictions."""
    from predict import main
    main()


def evaluate_results(args):
    """Evaluate results."""
    from evaluate import analyze_results
    if args.results_path:
        analyze_results(args.results_path)
    else:
        print("Error: --results-path required for evaluate command")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Multimodal Fusion Model CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s verify              # Check data structure
  %(prog)s test                # Test model build
  %(prog)s train               # Run training
  %(prog)s predict             # Generate predictions
  %(prog)s evaluate results.csv # Analyze results

For more help on each command, see README.md
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Verify command
    subparsers.add_parser('verify', help='Verify data structure')
    
    # Test command
    subparsers.add_parser('test', help='Test model architecture')
    
    # Train command
    subparsers.add_parser('train', help='Train the model with k-fold CV')
    
    # Predict command
    subparsers.add_parser('predict', help='Generate predictions for test set')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model results')
    eval_parser.add_argument('results_path', nargs='?', help='Path to results CSV')
    
    args = parser.parse_args()
    
    if args.command == 'verify':
        verify_data()
    elif args.command == 'test':
        test_model()
    elif args.command == 'train':
        train_model()
    elif args.command == 'predict':
        predict()
    elif args.command == 'evaluate':
        evaluate_results(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
