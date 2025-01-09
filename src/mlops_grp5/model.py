import torch
import torch.nn as nn
import pytorch_lightning as pl

class PlaceHolderModel(pl.LightningModule):
    """
    Placeholder model class for setting up the training pipeline
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28*28, 10)

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x
    
    def training_step(self, batch):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        return loss
    
    def test_step(self, batch):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == "__main__":
    model = PlaceHolderModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 28*28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")