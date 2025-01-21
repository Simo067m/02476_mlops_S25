import os

import hydra

from src.mlops_grp5.dataloaders import get_fruits_and_vegetables_dataloaders


def test_get_fruits_and_vegetables_dataloaders():
    hydra.initialize(config_path=os.path.join("..", "configs"), version_base="1.1")
    train_loader, test_loader, val_loader = get_fruits_and_vegetables_dataloaders()
    
    assert len(train_loader) > 0, "Train dataloader is empty"
    assert len(test_loader) > 0, "Test dataloader is empty"
    assert len(val_loader) > 0, "Validation dataloader is empty"
    assert train_loader.dataset[0][0].shape == (3, 224, 224), "Incorrect shape of training data"