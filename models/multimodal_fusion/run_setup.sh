#!/bin/bash
# Complete setup and run script for Multimodal Fusion Model
# This script will set up everything and prepare for training

set -e  # Exit on any error

echo "=========================================="
echo "🌱 Multimodal Fusion Model Setup"
echo "=========================================="

# Step 1: Check Python version
echo ""
echo "📌 Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Step 2: Create virtual environment
echo ""
echo "📌 Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
echo "✓ Virtual environment created"

# Step 3: Activate virtual environment
echo ""
echo "📌 Step 3: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Step 4: Upgrade pip
echo ""
echo "📌 Step 4: Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"

# Step 5: Install dependencies
echo ""
echo "📌 Step 5: Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Step 6: Verify installation
echo ""
echo "📌 Step 6: Verifying installation..."
python -c "import torch; print('  ✓ PyTorch:', torch.__version__)"
python -c "import timm; print('  ✓ TIMM:', timm.__version__)"
python -c "import albumentations; print('  ✓ Albumentations:', albumentations.__version__)"
python -c "import torch; print('  ✓ CUDA available:', torch.cuda.is_available())"

# Step 7: Create necessary directories
echo ""
echo "📌 Step 7: Creating directories..."
mkdir -p data/train/rgb data/train/depth data/train/masks
mkdir -p data/test/rgb data/test/depth
mkdir -p output/checkpoints output/logs output/evaluation
echo "✓ Directories created"

# Step 8: Check for existing data
echo ""
echo "📌 Step 8: Checking for existing data..."
if [ -d "../../datasets/Training" ]; then
    echo "Found existing training data in ../../datasets/Training"
    
    # Ask user if they want to copy/link data
    echo ""
    echo "Would you like to use this data? Options:"
    echo "  1) Create symbolic links (recommended, saves space)"
    echo "  2) Copy files (uses more space but safer)"
    echo "  3) Skip (I'll set it up manually)"
    read -p "Enter choice (1/2/3): " choice
    
    if [ "$choice" = "1" ]; then
        echo "Creating symbolic links..."
        ln -sf $(pwd)/../../datasets/Training/RGBImages/* data/train/rgb/
        ln -sf $(pwd)/../../datasets/Training/DepthImages/* data/train/depth/
        
        # Check for labels file
        if [ -f "../../datasets/Training/Train.csv" ]; then
            cp ../../datasets/Training/Train.csv data/train/labels.csv
            echo "✓ Labels file copied"
            
            # Verify CSV has correct columns
            python3 << 'PYEOF'
import pandas as pd
df = pd.read_csv('data/train/labels.csv')
print(f"  CSV columns: {df.columns.tolist()}")
if 'id' not in df.columns or 'dry_weight' not in df.columns:
    print("  ⚠️  Warning: CSV should have 'id' and 'dry_weight' columns")
    print("     Please rename columns if needed")
else:
    print(f"  ✓ CSV format looks good ({len(df)} samples)")
PYEOF
        fi
        
        echo "✓ Symbolic links created"
        
    elif [ "$choice" = "2" ]; then
        echo "Copying files (this may take a while)..."
        cp ../../datasets/Training/RGBImages/* data/train/rgb/
        cp ../../datasets/Training/DepthImages/* data/train/depth/
        if [ -f "../../datasets/Training/Train.csv" ]; then
            cp ../../datasets/Training/Train.csv data/train/labels.csv
        fi
        echo "✓ Files copied"
    else
        echo "Skipping data setup. Please organize manually."
    fi
else
    echo "⚠️  No existing data found. Please organize your data in:"
    echo "    data/train/rgb/"
    echo "    data/train/depth/"
    echo "    data/train/labels.csv"
fi

# Step 9: Test model build
echo ""
echo "📌 Step 9: Testing model build..."
python test_model.py

# Step 10: Summary
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📊 Next steps:"
echo ""
echo "1. If not done already, ensure your data is in place:"
echo "   - data/train/rgb/       (RGB images)"
echo "   - data/train/depth/     (Depth images)"
echo "   - data/train/labels.csv (id, dry_weight)"
echo ""
echo "2. Verify data structure:"
echo "   python verify_data.py"
echo ""
echo "3. Start training:"
echo "   python train.py"
echo ""
echo "4. After training, generate predictions:"
echo "   python predict.py"
echo ""
echo "=========================================="
echo "⚠️  Remember to activate the virtual environment:"
echo "   source venv/bin/activate"
echo "=========================================="
