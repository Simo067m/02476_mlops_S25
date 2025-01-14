import pytorch_lightning as pl
import torch
import torch.nn as nn

import timm
from dataloaders import get_fruits_and_vegetables_dataloaders

def get_model(model_name: str, num_classes: int) -> nn.Module:
    """Gets a pretrained image model from the timm library."""
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

class ImageModel(pl.LightningModule):
    """
    Defines an image model for PyTorch Lightning.
    Inherits from pl.LightningModule to leverage its functionality.
    """
    def __init__(self, learning_rate: float, weight_decay: float,
                 model_name: str = "test_efficientnet.r160_in1k") -> None:
        super().__init__()
        # Load pretrained model
        print(f"Initializing model {model_name}...")
        self.model = get_model(model_name, num_classes=2)

        # Define hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize loss and accuracy metrics
        self.train_epoch_loss = 0.0
        self.val_epoch_loss = 0.0
        self.test_epoch_loss = 0.0
        self.test_epoch_acc = 0.0

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.test_accs = []

        print("Model initialized.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning."""
        images, targets = batch
        pred = self(images)
        loss = self.loss_fn(pred, targets)
        acc = (pred.argmax(dim=-1) == targets).float().mean()
        self.train_epoch_loss = loss.item()
        self.train_losses.append(loss.item())
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step for PyTorch Lightning."""
        images, targets = batch
        pred = self(images)
        loss = self.loss_fn(pred, targets)
        self.val_epoch_loss = loss.item()
        self.val_losses.append(loss.item())
        self.log("val_loss", loss)
        self.log("val_acc", (pred.argmax(dim=-1) == targets).float().mean())
        return loss
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step for PyTorch Lightning."""
        images, targets = batch
        pred = self(images)
        loss = self.loss_fn(pred, targets)
        acc = (pred.argmax(dim=1) == targets).float().mean()
        self.test_epoch_loss = loss.item()
        self.test_epoch_acc = acc.item()
        self.test_losses.append(loss.item())
        self.test_accs.append(acc.item())
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    def on_train_epoch_start(self):
        self.train_epoch_loss = 0.0
        self.val_epoch_loss = 0.0
        self.test_epoch_loss = 0.0
    
    def on_train_epoch_end(self):
        print(f"Epoch {self.current_epoch + 1} Training Loss: {self.train_epoch_loss:.4f}")
    
    def on_validation_epoch_end(self):
        print(f"Epoch {self.current_epoch + 1} Validation Loss: {self.val_epoch_loss:.4f}")
    
    def on_test_epoch_end(self):
        print(f"Test Loss: {self.test_epoch_loss:.4f} Test Accuracy: {self.test_epoch_acc:.4f}")

if __name__ == "__main__":
    # Initialize model
    model_name = "test_efficientnet.r160_in1k"
    model = ImageModel(learning_rate=1e-3, weight_decay=1e-5, model_name=model_name)

    # Load data
    _, test_loader, _ = get_fruits_and_vegetables_dataloaders()

    # Perform forward pass
    outputs = model(test_loader.dataset[0][0].unsqueeze(0))

    # Print model outputs
    print(outputs)
    print(outputs.shape)