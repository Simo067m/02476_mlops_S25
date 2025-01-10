import torch
from model import PlaceHolderModel
from pytorch_lightning import Trainer
from visualize import plot_placeholder_loss

from data import get_placeholder_dataloader

if __name__ == "__main__":
    trainer = Trainer(max_epochs=10, accelerator="gpu", devices=1, logger=False, enable_checkpointing=False)
    model = PlaceHolderModel()
    train_loader, test_loader = get_placeholder_dataloader()
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)
    print("Training and testing complete.")
    torch.save(model.state_dict(), "models/model.pth")
    print("Model saved.")
    plot_placeholder_loss()
    print("Loss plot saved.")