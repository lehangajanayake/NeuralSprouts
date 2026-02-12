import torch

from models.model_v8.model import LettuceSAMFusionNet


def test_forward_shapes():
    model = LettuceSAMFusionNet()
    rgb = torch.randn(2, 3, 96, 96)
    rgbd = torch.randn(2, 4, 96, 96)

    rgb_pred, rgbd_pred, fusion_pred = model(rgb, rgbd)
    assert rgb_pred.shape == (2,)
    assert rgbd_pred.shape == (2,)
    assert fusion_pred.shape == (2,)
