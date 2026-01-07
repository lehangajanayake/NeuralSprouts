import torch

from models.model_v4.model import LettuceMultiBranchCNN


def test_forward_shapes():
    model = LettuceMultiBranchCNN(num_classes=4)
    rgb = torch.randn(2, 3, 64, 64)
    rgbd = torch.randn(2, 4, 64, 64)

    logits, rgbd_pred, fusion_pred = model(rgb, rgbd)
    assert logits.shape == (2, 4)
    assert rgbd_pred.shape == (2,)
    assert fusion_pred.shape == (2,)
