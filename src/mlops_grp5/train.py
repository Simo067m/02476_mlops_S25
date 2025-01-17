import os

import pytorch_lightning as pl
import torch
import typer
from dataloaders import get_fruits_and_vegetables_dataloaders
from model import ImageModel
from pytorch_lightning import Trainer
from visualize import plot_accuracy, plot_loss

import wandb

from logger import log

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(batch_size: int = 32, learning_rate: float = 1e-3, weight_decay: float = 1e-5, max_epochs: int = 10):
    log.info(f"Training on device: {DEVICE}")
    torch.cuda.empty_cache()
    log.info(('Training model with batch size:', batch_size, 'learning rate:', learning_rate, 'and weight decay:', weight_decay))
    #wait for input in terminal
    # Define the model
    model = ImageModel(learning_rate=learning_rate, weight_decay=weight_decay).to(DEVICE)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min')
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')
    # Define the trainer
    trainer = Trainer(max_epochs=max_epochs, accelerator=DEVICE, devices=1, logger=pl.loggers.WandbLogger(project='mlops-grp5', config={"batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay}, log_model=True), callbacks=[checkpoint_callback, early_stop_callback])

    # Load data
    train_loader, test_loader, val_loader = get_fruits_and_vegetables_dataloaders()

    # Train and test the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    log.info("Training and testing complete.")
    
    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/model.pth")
    log.info("Model saved.")

    # Save the loss plot
    plot_path = "reports/figures"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plot_loss(model.train_losses, model.val_losses, model.test_losses, plot_path)
    plot_accuracy(model.test_accs, plot_path)
    log.info("Visualizations saved.")
    wandb.finish()

if __name__ == "__main__":
    typer.run(train_model)
