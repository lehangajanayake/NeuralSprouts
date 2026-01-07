# NeuralSprouts

AI models for predicting lettuce dry weight from RGB and depth images.

## Structure

```
NeuralSprouts/
├── datasets/          # Training and test datasets with RGB/Depth images
└── models/            # Multiple model versions with different architectures
    ├── simple_cnn/           # Baseline lightweight CNN
    ├── model_v1/             # Multi-branch CNN (RGBD + RGB)
    ├── model_v2/             # Three-branch CNN (adds leaf area)
    ├── model_v3/             # ResNet18 transfer learning
    ├── model_v4/             # Fusion network with staged training
    └── multimodal_fusion/    # Advanced architecture with segmentation
```

## Getting Started

1. Add your dataset to the `datasets/` folder
2. Each model is in its own folder under `models/` with independent training scripts
3. See individual model README files for specific setup and training instructions

## Models

### Model Evolution

- **simple_cnn**: Lightweight baseline CNN for quick prototyping and baseline performance
  - Single-task regression (dry weight only)
  - 3 convolutional layers, simple architecture
  
- **model_v1**: Multi-branch CNN with dual tasks
  - RGBD branch for dry weight regression
  - RGB branch for variety classification
  - 6-layer and 5-layer convolutional networks

- **model_v2**: Extended three-branch architecture
  - Adds depth-only branch for leaf area prediction
  - Three separate task objectives
  - Enhanced multi-task learning

- **model_v3**: Transfer learning with ResNet18
  - Pretrained ImageNet weights
  - Single-task regression focus
  - Includes Streamlit debugging tools

- **model_v4**: Advanced fusion network
  - RGB classification + RGBD regression branches
  - Fusion network combines features
  - Three-stage training strategy

- **multimodal_fusion**: State-of-the-art architecture
  - ConvNeXt/EfficientNet backbones
  - Semantic segmentation + regression
  - Phenotype feature extraction
  - K-fold cross-validation and ensemble inference

Each model folder contains its own README with detailed architecture descriptions, usage instructions, and training guides. 
