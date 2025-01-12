import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.io

import matplotlib.pyplot as plt
import numpy as np


class PlaceHolderDataset(Dataset):
    """
    Placeholder dataset class for setting up the dataloading pipeline
    data_path is unused in this example, but is included to show that it can be used in the CLI
    """

    def __init__(self, data_path: Path, data_len: int = 50000):
        print(f"Entered data path: {data_path}")
        # self.data = torch.randn(data_len, 28*28)
        # self.targets = torch.randint(0, 10, (data_len,))

        self.data_path = data_path
        self.image_paths = []
        self.labels = []

        # Define class-to-index mapping
        self.class_to_idx = {"Fresh": 0, "Rotten": 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Walk through the data directory and find all images
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):  # Check for image files
                    file_path = os.path.join(root, file)
                    self.image_paths.append(file_path)
                    # Determine label based on filename
                    if "fresh" in file.lower():
                        self.labels.append(self.class_to_idx["Fresh"])  # 0 for 'Fresh'
                    elif "rotten" in file.lower():
                        self.labels.append(self.class_to_idx["Rotten"])  # 1 for 'Rotten'
                    else:
                        raise ValueError(f"File {file} does not specify 'fresh' or 'rotten'.")

    def __len__(self):
        # return len(self.data)
        return len(self.image_paths)

    def __getitem__(self, idx):
        # return self.data[idx], self.targets[idx]
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = torchvision.io.read_image(image_path).float() / 255.0

        return image, label

    def preprocess(self, output_folder: Path):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        #torch.save(self.data, output_folder / "data.pt")
        #torch.save(self.targets, output_folder / "targets.pt")


def get_placeholder_dataloader(
    data_len: int = 5000, batch_size: int = 32, image_size: int = 28 * 28, train_split: float = 0.8
):
    """Returns dataloaders for the placeholder dataset."""
    data_path = Path("data/Fruits_Vegetables_Dataset(12000)")

    # Define transforms for the dataset
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = PlaceHolderDataset(data_path, data_len=data_len)

    dataset_len = len(dataset)
    train_size = int(train_split * dataset_len)
    test_size = dataset_len - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # train_dataset = PlaceHolderDataset(data_path, data_len=data_len)
    # test_dataset = PlaceHolderDataset(data_path, data_len=int(data_len * 0.25))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_placeholder_dataloader()

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # Print class-to-index mapping
    print("Class-to-index mapping:", train_dataset.dataset.class_to_idx)

    # Print shapes of training and test datasets
    print("Training dataset shape:", len(train_loader.dataset))
    print("Test dataset shape:", len(test_loader.dataset))

    # Printing shape of total dataset
    print("Total dataset shape:", len(train_loader.dataset) + len(test_loader.dataset))

    # Printing size of first image
    sample_image, sample_label = train_dataset[4]
    print(f"Sample image shape: {sample_image.size()}")

"""    # Function to count fresh and rotten images
    def count_classes(dataset):
        fresh_count = 0
        rotten_count = 0

        for idx in range(len(dataset)):
            try:
                _, label = dataset[idx]  # Try to access the label
                if label == 0:  # Fresh
                    fresh_count += 1
                elif label == 1:  # Rotten
                    rotten_count += 1
            except Exception as e:
                print(f"Skipping image at index {idx} due to error: {e}")

        return fresh_count, rotten_count

    # Count classes in train and test datasets
    train_fresh_count, train_rotten_count = count_classes(train_dataset)
    test_fresh_count, test_rotten_count = count_classes(test_dataset)

    print(f"Number of fresh images in training dataset: {train_fresh_count}")
    print(f"Number of rotten images in training dataset: {train_rotten_count}")
    print(f"Number of fresh images in test dataset: {test_fresh_count}")
    print(f"Number of rotten images in test dataset: {test_rotten_count}")"""
