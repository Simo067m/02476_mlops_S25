import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class PlaceHolderDataset(Dataset):
    """
    Placeholder dataset class for setting up the dataloading pipeline
    data_path is unused in this example, but is included to show that it can be used in the CLI
    """
    def __init__(self, data_path: Path, data_len: int = 50000):
        print(f"Entered data path: {data_path}")
        self.data = torch.randn(data_len, 28*28)
        self.targets = torch.randint(0, 10, (data_len,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def preprocess(self, output_folder: Path):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        torch.save(self.data, output_folder / "data.pt")
        torch.save(self.targets, output_folder / "targets.pt")

def get_placeholder_dataloader(data_len: int = 5000, batch_size: int = 32):
    """Returns dataloaders for the placeholder dataset."""
    data_path = Path("data/")
    train_dataset = PlaceHolderDataset(data_path, data_len=data_len)
    test_dataset = PlaceHolderDataset(data_path, data_len=int(data_len * 0.25))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_placeholder_dataloader()