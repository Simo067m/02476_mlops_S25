import torch
import os
from model import PlaceHolderModel
from pytorch_lightning import Trainer
from visualize import plot_placeholder_loss

from dataloaders import get_fruits_and_vegetables_dataloaders

if __name__ == "__main__":
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(max_epochs=10, accelerator=device, devices=1, logger=False, enable_checkpointing=False)
    train_loader, test_loader = get_fruits_and_vegetables_dataloaders()
    """
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)
    print("Training and testing complete.")
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/model.pth")
    print("Model saved.")
    if not os.path.exists("reports/figures"):
        os.makedirs("reports/figures")
    plot_placeholder_loss()
    print("Loss plot saved.")
    """