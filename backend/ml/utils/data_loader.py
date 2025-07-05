import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define constants for our models
IMAGE_SIZE = 224  # A standard size for many pretrained models like ResNet/ViT
BATCH_SIZE = 32   # Number of images to process in one go
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# 1. Transformations for the training set (with data augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(), # Converts image to a PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard normalization
])

# 2. Transformations for the validation/testing set (only resizing and normalization)
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TbiDataset(Dataset):
    """Custom PyTorch Dataset for loading TBI images."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Create a mapping from class name (e.g., 'subdural') to an integer (e.g., 5)
        self.class_to_idx = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}
        self.image_paths, self.labels = self._load_dataset()

    # def _load_dataset(self):
    #     """Walks through the data directory to find all images and their labels."""
    #     image_paths = []
    #     labels = []
    #     # Loop through each folder (e.g., 'data/epidural/')
    #     for class_name in os.listdir(self.data_dir):
    #         class_dir = os.path.join(self.data_dir, class_name)
    #         if os.path.isdir(class_dir) and class_name in self.class_to_idx:
    #             # Loop through each image in the folder
    #             for img_name in os.listdir(class_dir):
    #                 # Add the full image path and its corresponding integer label
    #                 image_paths.append(os.path.join(class_dir, img_name))
    #                 labels.append(self.class_to_idx[class_name])
    #     return image_paths, labels
    def _load_dataset(self):
        """Walks through the data directory to find a subset of images and their labels."""
        image_paths = []
        labels = []
    # Loop through each folder (e.g., 'data/epidural/')
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir) and class_name in self.class_to_idx:
            # Loop through the first 500 images in the folder
                for img_name in os.listdir(class_dir)[:500]: # MODIFIED: Only take the first 500
                # Add the full image path and its corresponding integer label
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(self.class_to_idx[class_name])
        return image_paths, labels

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Fetches a single image and its label by index."""
        img_path = self.image_paths[idx]
        # Open the image file and ensure it's in RGB format
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # Apply the transformations (e.g., resizing, augmenting)
        if self.transform:
            image = self.transform(image)

        return image, label