import torch
import os
from model import ImageModel
from pytorch_lightning import Trainer
from visualize import plot_placeholder_loss
import wandb
import pytorch_lightning as pl
import typer

from dataloaders import get_fruits_and_vegetables_dataloaders
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(learning_rate: float = 1e-3, weight_decay: float = 1e-5, max_epochs: int = 10):
    print(f"Training on device:", DEVICE)
    torch.cuda.empty_cache()
    print('Training model with learning rate:', learning_rate, 'and weight decay:', weight_decay)
    print('Learning_rate type:', type(learning_rate), 'Weight_decay type:', type(weight_decay))
    #wait for input in terminal
    # Define the model
    model = ImageModel(learning_rate=learning_rate, weight_decay=weight_decay).to(DEVICE)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min')
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')
    # Define the trainer
    trainer = Trainer(max_epochs=max_epochs, accelerator=DEVICE, devices=1, logger=pl.loggers.WandbLogger(project='mlops-grp5', config={"learning_rate": learning_rate, "weight_decay": weight_decay}, log_model=True), callbacks=[checkpoint_callback, early_stop_callback])

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
    wandb.finish()

if __name__ == "__main__":
    typer.run(train_model)
