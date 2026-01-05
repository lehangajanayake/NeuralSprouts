#!/bin/bash
# Setup script for Multimodal Fusion Model

echo "=========================================="
echo "Multimodal Fusion Model Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Warning: Python 3.10+ recommended. You have $python_version"
fi

# Create virtual environment (optional)
echo ""
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
    echo ""
    echo "To activate:"
    echo "  source venv/bin/activate  # On macOS/Linux"
    echo "  venv\\Scripts\\activate     # On Windows"
    echo ""
    read -p "Activate now and continue? (y/n): " activate_now
    if [ "$activate_now" = "y" ]; then
        source venv/bin/activate
    fi
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Error installing dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating output directories..."
mkdir -p output/checkpoints
mkdir -p output/logs
mkdir -p output/evaluation
mkdir -p data/train/rgb
mkdir -p data/train/depth
mkdir -p data/train/masks
mkdir -p data/test/rgb
mkdir -p data/test/depth

echo "✓ Directories created"

# Test installation
echo ""
echo "Testing installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || exit 1
python -c "import timm; print(f'TIMM: {timm.__version__}')" || exit 1
python -c "import albumentations; print(f'Albumentations: {albumentations.__version__}')" || exit 1

echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Test model build
echo ""
read -p "Test model build? (y/n): " test_model
if [ "$test_model" = "y" ]; then
    echo "Testing model build..."
    python test_model.py
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare your data in data/train/ directory"
echo "2. Run: python verify_data.py"
echo "3. Run: python train.py"
echo ""
echo "For detailed instructions, see:"
echo "  - QUICK_START.md"
echo "  - README.md"
echo ""
echo "=========================================="
