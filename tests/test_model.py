import torch

from src.mlops_grp5.model import ImageModel


def test_image_model():
    model = ImageModel(1., 1.)
    
    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)

    assert output.shape == (1, 2), f"Expected output shape (1, 2), but got {output.shape}"