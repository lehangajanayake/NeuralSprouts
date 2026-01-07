"""
Test model building and forward pass.

Quick script to verify the model architecture works correctly.
"""

import torch
from config import Config
from model import build_model


def test_model_build():
    """Test that model can be built."""
    print("="*60)
    print("Testing Model Build")
    print("="*60)
    
    try:
        print("\nBuilding model...")
        config = Config()
        model = build_model(config)
        print("✓ Model built successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / (1024**2):.2f} MB (fp32)")
        
        return model
    except Exception as e:
        print(f"✗ Error building model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)
    
    try:
        config = Config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        model = model.to(device)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        rgb = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
        depth = torch.randn(batch_size, 1, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
        
        print(f"\nInput shapes:")
        print(f"  RGB: {rgb.shape}")
        print(f"  Depth: {depth.shape}")
        
        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(rgb, depth)
        
        print("✓ Forward pass successful")
        
        print(f"\nOutput shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        # Check outputs
        assert 'final_pred' in outputs, "Missing final_pred in outputs"
        assert 'deep_pred' in outputs, "Missing deep_pred in outputs"
        
        if config.USE_PHENOTYPE_FEATURES:
            assert 'mask_logits' in outputs, "Missing mask_logits in outputs"
            assert 'phen_pred' in outputs, "Missing phen_pred in outputs"
            assert 'phenotype_features' in outputs, "Missing phenotype_features in outputs"
            print(f"\n✓ All expected outputs present (with phenotype features)")
        else:
            print(f"\n✓ All expected outputs present (without phenotype features)")
        
        return True
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model):
    """Test backward pass and gradient computation."""
    print("\n" + "="*60)
    print("Testing Backward Pass")
    print("="*60)
    
    try:
        config = Config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = model.to(device)
        model.train()
        
        # Create dummy inputs and targets
        batch_size = 2
        rgb = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
        depth = torch.randn(batch_size, 1, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
        target_weight = torch.randn(batch_size).to(device)
        
        # Forward pass
        outputs = model(rgb, depth)
        
        # Simple loss
        loss = torch.nn.functional.mse_loss(outputs['final_pred'], target_weight)
        
        print(f"\nLoss: {loss.item():.6f}")
        
        # Backward pass
        print("Running backward pass...")
        loss.backward()
        
        # Check gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        if has_gradients:
            print("✓ Gradients computed successfully")
        else:
            print("✗ No gradients found")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Model Architecture Test Suite")
    print("="*60)
    
    # Test 1: Build model
    model = test_model_build()
    if model is None:
        print("\n✗ Model build failed. Cannot proceed with other tests.")
        return
    
    # Test 2: Forward pass
    forward_ok = test_forward_pass(model)
    if not forward_ok:
        print("\n✗ Forward pass failed.")
        return
    
    # Test 3: Backward pass
    backward_ok = test_backward_pass(model)
    if not backward_ok:
        print("\n✗ Backward pass failed.")
        return
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✓ Model build: PASSED")
    print("✓ Forward pass: PASSED")
    print("✓ Backward pass: PASSED")
    print("\n🎉 All tests passed! Model is ready for training.")
    print("="*60)


if __name__ == '__main__':
    main()
