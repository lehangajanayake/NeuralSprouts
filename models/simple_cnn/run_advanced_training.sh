#!/bin/bash

# Quick Start Script for Advanced Lettuce Model Training
# Usage: ./run_advanced_training.sh

echo "=================================================="
echo "   Advanced Lettuce Dry Weight Prediction Model"
echo "=================================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "advanced_train.py" ]; then
    echo "Error: Please run this script from models/simple_cnn directory"
    echo "Usage: cd models/simple_cnn && ./run_advanced_training.sh"
    exit 1
fi

# Check if datasets exist
if [ ! -d "../../datasets/Training/RGBImages" ]; then
    echo "Error: Training RGB images not found!"
    echo "Expected: ../../datasets/Training/RGBImages"
    exit 1
fi

if [ ! -d "../../datasets/Training/DepthImages" ]; then
    echo "Error: Training Depth images not found!"
    echo "Expected: ../../datasets/Training/DepthImages"
    exit 1
fi

if [ ! -f "../../datasets/Training/Train.csv" ]; then
    echo "Error: Training labels not found!"
    echo "Expected: ../../datasets/Training/Train.csv"
    exit 1
fi

echo "✓ Dataset files found"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import torch; import torchvision; import pandas; import sklearn; import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r ../../requirements.txt
else
    echo "✓ All dependencies installed"
fi

echo ""
echo "Starting training..."
echo "=================================================="
echo ""

# Run the advanced training script
python3 advanced_train.py

echo ""
echo "=================================================="
echo "Training complete! Check the output above for results."
echo "=================================================="
