import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import LettuceNormalFusionNet


def test_forward_shapes():
    model = LettuceNormalFusionNet(drop_path_prob=0.0)
    rgbn = torch.randn(2, 6, 96, 96)
    rgbd = torch.randn(2, 4, 96, 96)

    rgbn_pred, rgbd_pred, fusion_pred = model(rgbn, rgbd)

    assert rgbn_pred.shape == (2,)
    assert rgbd_pred.shape == (2,)
    assert fusion_pred.shape == (2,)
