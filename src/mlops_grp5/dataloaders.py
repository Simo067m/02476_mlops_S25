import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import kagglehub

# Seed for reproducibility
torch.manual_seed(0)

class FruitsVegetablesDataset(Dataset):
    """
    Dataset class for the fruits and vegetables dataset.
    """

    def __init__(self, data_path: Path):
        print(f"Data path: {data_path}")
        
        self.data_path = data_path
        self.save_data_path = self.data_path / "processed_data"
        
        # Define class-to-index mapping
        self.class_to_idx = {"Fresh": 0, "Rotten": 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Check if data should be processed, if not, load the data
        if not os.path.exists(self.save_data_path):
            self.pre_process()
        else:
            print("Loading pre-processed data...")
            self.data = torch.load(self.save_data_path / "data.pt", weights_only=True)
            self.labels = torch.load(self.save_data_path / "labels.pt", weights_only=True)
        
        print("Data loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]
    
    def pre_process(self):
        """Loads the images into tensors and performs transformations on them."""
        print("Pre-processing data...")
        self.labels = []
        image_paths = []

        # Walk through the data directory and find all images
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):  # Check for image files
                    file_path = os.path.join(root, file)
                    image_paths.append(file_path)
                    # Determine label based on filename
                    if "fresh" in file.lower():
                        self.labels.append(self.class_to_idx["Fresh"])  # 0 for 'Fresh'
                    elif "rotten" in file.lower():
                        self.labels.append(self.class_to_idx["Rotten"])  # 1 for 'Rotten'
                    else:
                        raise ValueError(f"File {file} does not specify 'fresh' or 'rotten'.")

        # Load images and apply transformations

        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize(256),                 # Resize the image
            transforms.CenterCrop(224),             # Crop to 224x224 pixels
            transforms.ToTensor(),                  # Convert to a tensor
            transforms.Normalize(                   # Normalize using ImageNet's mean and std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        tensor_list = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")  # Load image and ensure RGB mode
            tensor = transform(image)  # Apply transformations
            tensor_list.append(tensor)
        
        # Stack tensors into a single tensor with shape [n_images, 3, height, width]
        self.data = torch.stack(tensor_list)

        # Save the processed data
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path, exist_ok=True)
        torch.save(self.data, self.save_data_path / "data.pt")
        torch.save(self.labels, self.save_data_path / "labels.pt")
        print("Data pre-processing complete.")


def get_fruits_and_vegetables_dataloaders(
    batch_size: int = 32, train_split: float = 0.6, test_split: float = 0.2
):
    """Returns dataloaders for the fruits and vegetables dataset."""
    data_path = Path("data/fruits_vegetables_dataset")

    dataset = FruitsVegetablesDataset(data_path)

    dataset_len = len(dataset)
    train_size = int(train_split * dataset_len)
    test_size = int(test_split * dataset_len)
    val_size = dataset_len - train_size - test_size

    # Split into train and test
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    return train_loader, test_loader, val_loader

def download_fruits_and_vegetables_dataset():
    """Downloads the fruits and vegetables dataset using KaggleHub."""
    # Download latest version
    path = kagglehub.dataset_download("muhriddinmuxiddinov/fruits-and-vegetables-dataset")

    print("Path to dataset files:", path)


if __name__ == "__main__":
    train_loader, test_loader, val_loader = get_fruits_and_vegetables_dataloaders()

    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    val_dataset = val_loader.dataset

    # Print class-to-index mapping
    print("Class-to-index mapping:", train_dataset.dataset.class_to_idx)

    # Print lengths of training and test datasets
    print("Training dataset length:", len(train_loader.dataset))
    print("Test dataset length:", len(test_loader.dataset))
    print("Validation dataset length:", len(val_loader.dataset))

    # Printing length of total dataset
    print("Total dataset length:", len(train_dataset) + len(test_dataset), + len(val_dataset))

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
