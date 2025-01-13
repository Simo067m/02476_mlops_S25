import pytorch_lightning as pl
import torch
import torch.nn as nn

import timm
from torchvision import transforms
from PIL import Image

def get_model(model_name: str, num_classes: int) -> nn.Module:
    """Gets a pretrained image model from the timm library."""
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def model_forward(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Makes a prediction using a model."""
    # Make predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        outputs = model(imgs)
    
    return outputs

if __name__ == "__main__":
    model_name = "resnet50_clip.openai"
    model = get_model(model_name, num_classes=2)

    image = Image.open("data/fruits_vegetables_dataset/Fruits/FreshApple/freshApple (1).jpg")

    outputs = model_forward(model, image)

    print(outputs)
    print(outputs.shape)