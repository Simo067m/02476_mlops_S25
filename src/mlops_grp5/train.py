import os

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer

import wandb
from mlops_grp5.dataloaders import get_fruits_and_vegetables_dataloaders
from mlops_grp5.logger import log
from mlops_grp5.model import ImageModel
from mlops_grp5.visualize import plot_accuracy, plot_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs", version_base="1.1")
def train_model(config):
    """Trains and tests a model."""
    log.info(f"Training on device: {DEVICE}")
    torch.cuda.empty_cache()
    log.info(f"Using config: {config}")
    log.info(('Training model with batch size:', config.hyperparameters.batch_size, 'learning rate:', config.optimizer.learning_rate, 'and weight decay:', config.optimizer.weight_decay))
    #wait for input in terminal
    # Define the model
    model = ImageModel(learning_rate=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay).to(DEVICE)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min')
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')
    # Define the trainer
    trainer = Trainer(max_epochs=config.hyperparameters.max_epochs, accelerator=DEVICE, devices=1, logger=pl.loggers.WandbLogger(project='mlops-grp5', config={"batch_size": config.hyperparameters.batch_size, "learning_rate": config.optimizer.learning_rate, "weight_decay": config.optimizer.weight_decay}, log_model=True), callbacks=[checkpoint_callback, early_stop_callback])

    # Load data
    train_loader, test_loader, val_loader = get_fruits_and_vegetables_dataloaders(batch_size=config.hyperparameters.batch_size)

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
    train_model()
