# Virtual Environment Setup Guide

## Quick Answer for C Shell (.csh)

```csh
# Activate virtual environment in C shell
source venv/bin/activate.csh
```

---

## Complete Setup Instructions (All Shells)

### Step 1: Navigate to Project Directory
```bash
cd /Users/hansikodikara/NeuralSprouts/models/multimodal_fusion
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 3: Activate Virtual Environment

**Choose based on your shell:**

#### macOS/Linux - Bash/Zsh (most common)
```bash
source venv/bin/activate
```

#### C Shell (csh/tcsh)
```csh
source venv/bin/activate.csh
```

#### Fish Shell
```fish
source venv/bin/activate.fish
```

#### Windows - PowerShell
```powershell
venv\Scripts\Activate.ps1
```

#### Windows - Command Prompt
```cmd
venv\Scripts\activate.bat
```

---

## How to Check Which Shell You're Using

```bash
echo $SHELL
```

**Output examples:**
- `/bin/bash` → Use `source venv/bin/activate`
- `/bin/zsh` → Use `source venv/bin/activate`
- `/bin/csh` or `/bin/tcsh` → Use `source venv/bin/activate.csh`
- `/bin/fish` → Use `source venv/bin/activate.fish`

---

## Complete Setup & Run (Step-by-Step)

### For C Shell Users:

```csh
# 1. Navigate to project
cd /Users/hansikodikara/NeuralSprouts/models/multimodal_fusion

# 2. Create virtual environment (one time only)
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate.csh

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 7. Create data directories
mkdir -p data/train/rgb data/train/depth data/train/masks
mkdir -p data/test/rgb data/test/depth
mkdir -p output/checkpoints output/logs

# 8. Prepare your data (copy from existing datasets)
# Option A: Create symbolic links (saves space)
ln -s ../../datasets/Training/RGBImages/* data/train/rgb/
ln -s ../../datasets/Training/DepthImages/* data/train/depth/

# Option B: Copy files
# cp ../../datasets/Training/RGBImages/* data/train/rgb/
# cp ../../datasets/Training/DepthImages/* data/train/depth/

# 9. Copy labels file
cp ../../datasets/Training/Train.csv data/train/labels.csv

# 10. Verify data structure
python verify_data.py

# 11. (Optional) Test model build
python test_model.py

# 12. Start training
python train.py

# When done, deactivate virtual environment
deactivate
```

---

## For Bash/Zsh Users (Default on macOS):

```bash
# 1. Navigate to project
cd /Users/hansikodikara/NeuralSprouts/models/multimodal_fusion

# 2. Create virtual environment (one time only)
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Setup data directories
mkdir -p data/train/{rgb,depth,masks}
mkdir -p data/test/{rgb,depth}
mkdir -p output/{checkpoints,logs,evaluation}

# 6. Link or copy your existing data
ln -s $(pwd)/../../datasets/Training/RGBImages/* data/train/rgb/
ln -s $(pwd)/../../datasets/Training/DepthImages/* data/train/depth/
cp ../../datasets/Training/Train.csv data/train/labels.csv

# 7. Verify and run
python verify_data.py
python train.py
```

---

## Quick One-Liner Setup (Bash/Zsh)

```bash
cd /Users/hansikodikara/NeuralSprouts/models/multimodal_fusion && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt && \
echo "✓ Setup complete! Run 'python verify_data.py' then 'python train.py'"
```

---

## Quick One-Liner Setup (C Shell)

```csh
cd /Users/hansikodikara/NeuralSprouts/models/multimodal_fusion && \
python3 -m venv venv && \
source venv/bin/activate.csh && \
pip install --upgrade pip && \
pip install -r requirements.txt && \
echo "✓ Setup complete! Run 'python verify_data.py' then 'python train.py'"
```

---

## Verify Virtual Environment is Activated

You should see `(venv)` at the beginning of your prompt:

```
(venv) user@machine:~/NeuralSprouts/models/multimodal_fusion$
```

Or check:
```bash
which python
# Should show: /Users/hansikodikara/NeuralSprouts/models/multimodal_fusion/venv/bin/python
```

---

## Important Notes

1. **Always activate the virtual environment before running any Python commands**
   - `source venv/bin/activate.csh` (C shell)
   - `source venv/bin/activate` (Bash/Zsh)

2. **Data preparation**: Make sure your labels CSV has columns `id` and `dry_weight`
   ```bash
   # Check CSV format
   head -5 data/train/labels.csv
   ```

3. **If you get "No masks" warning**: That's OK! Just set in `config.py`:
   ```python
   USE_PHENOTYPE_FEATURES = False
   ```

4. **To deactivate virtual environment when done**:
   ```bash
   deactivate
   ```

---

## Troubleshooting

### "activate.csh not found"
```bash
# The venv wasn't created properly. Recreate it:
rm -rf venv
python3 -m venv venv
source venv/bin/activate.csh
```

### "Permission denied"
```bash
chmod +x venv/bin/activate.csh
source venv/bin/activate.csh
```

### "pip not found"
```bash
# Use python -m pip instead:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## After Setup, You'll Run:

```csh
# Every time you start a new terminal session:
cd /Users/hansikodikara/NeuralSprouts/models/multimodal_fusion
source venv/bin/activate.csh

# Then run your commands:
python train.py        # Train model (get MAE)
python predict.py      # Generate predictions
python evaluate.py     # Analyze results
```

---

**Bottom Line for C Shell:**
```csh
source venv/bin/activate.csh    # NOT just "activate.csh"
```

The `source` command is required to activate the environment!
