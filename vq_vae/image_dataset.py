import h5py
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        self.h5_file_path = h5_file_path
        self.transform = transform

        self.route_lengths = []

        with h5py.File(h5_file_path, 'r') as h5_file:
            for i in range(9999):
                dataset_name = f'images_{i}'
                if dataset_name in h5_file:
                    self.route_lengths.append(len(h5_file[dataset_name]))
                else:
                    break

    def __len__(self):
        return sum(self.route_lengths)

    def __getitem__(self, idx):
        route_number = 0
        for length in self.route_lengths:
            if idx >= length:
                route_number += 1
                idx -= length
            else:
                break

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            image = h5_file[f'images_{route_number}'][idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            pil_image = Image.fromarray(image)
            image = self.transform(pil_image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            image = torch.tensor(image, dtype=torch.float32)
        return image, route_number
