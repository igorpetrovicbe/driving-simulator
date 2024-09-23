import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, image_folder):
        """
        Args:
            image_folder (str): Path to the folder containing the images.
        """
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')  # Ensure it's in RGB format
        image = np.array(image)  # Convert to numpy array

        # Convert to PyTorch tensor and permute the axes
        image = torch.from_numpy(image).permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)
        image = torch.tensor(image, dtype=torch.float32)  # Ensure it's a float32 tensor

        return image
