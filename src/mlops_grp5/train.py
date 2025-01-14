import torch
import os
from model import ImageModel
from pytorch_lightning import Trainer
from visualize import plot_placeholder_loss

from dataloaders import get_fruits_and_vegetables_dataloaders
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(f"Training on device:", DEVICE)

    # Define the model
    model = ImageModel(learning_rate=1e-3, weight_decay=1e-5).to(DEVICE)

    # Define the trainer
    trainer = Trainer(max_epochs=10, accelerator=DEVICE, devices=1, logger=False, enable_checkpointing=False)

    # Load data
    train_loader, test_loader, val_loader = get_fruits_and_vegetables_dataloaders()

    # Train and test the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    print("Training and testing complete.")

    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/model.pth")
    print("Model saved.")

    # Save the loss plot
    if not os.path.exists("reports/figures"):
        os.makedirs("reports/figures")
    plot_placeholder_loss()
    print("Loss plot saved.")