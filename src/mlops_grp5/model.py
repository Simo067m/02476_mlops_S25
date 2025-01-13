import pytorch_lightning as pl
import torch
import torch.nn as nn

import timm

if __name__ == "__main__":
    model = timm.create_model("resnet50_clip.openai", pretrained=True, num_classes=2)
    model.eval()
    