import torch
from src.model import build_model

def test_forward():
    m = build_model('resnet18', 2, pretrained=False)
    x = torch.randn(2,3,224,224)
    y = m(x)
    assert y.shape == (2,2)