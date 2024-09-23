# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.
import os

import numpy as np
import torch

from PIL import Image
from torch.utils.data import DataLoader, Subset

from image_dataset import ImageDataset
from vqvae import VQVAE
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=160)

# Initialize model.
device = torch.device("cuda:0")
use_ema = True
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 3,
    "num_residual_layers": 2,
    "num_residual_hiddens": 128,
    "embedding_dim": 256,
    "num_embeddings": 1024,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}

if __name__ == '__main__':
    model = VQVAE(**model_args).to(device)

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    model.load_state_dict(torch.load('vqvae_v4_2999.pth'))

    #file_path = 'driving_images3.h5'
    file_path = 'D:\\datasets\\driving_images4.h5'

    print('h1')
    train_dataset = ImageDataset(file_path)
    print('h2')
    # Initialize the DataLoader (replace this with your actual DataLoader initialization)
    unshuffled_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    indices_list = []

    counter = 0
    vert_dim = 17
    hor_dim = 68

    current_video_number = -1

    # Iterate over the DataLoader and sample N consecutive data points
    model.eval()
    with torch.no_grad():
        for batch in unshuffled_loader:
            # Assuming each batch contains both inputs and targets
            inputs, video_numbers = batch

            #outputs = model(inputs.to(device))["x_recon"]

            quantizer = model.get_quantizer()
            encoding = model.encode(inputs.to(device))
            encoding_indices = quantizer.get_quantized_indices(encoding)
            test_indices = encoding_indices.view(inputs.shape[0], vert_dim, hor_dim)
            test_indices2 = test_indices.transpose(1, 2)
            flat_indices = test_indices2.reshape(-1)

            if video_numbers[0].item() != current_video_number:
                indices_list.append(flat_indices.tolist())
                current_video_number = video_numbers[0].item()
            else:
                indices_list[current_video_number].extend(flat_indices.tolist())

            if (counter + 1) % 10 == 0:
                print(f'{counter}/{len(unshuffled_loader)}')
            counter += 1

            #if current_video_number == 2:
            #    break

    # Open a file in write mode
    for i in range(len(indices_list)):
        with open(f'out_dataset/{i+1}_img.txt', 'w') as file:
            # Convert each integer to a string and join them with spaces
            content = '\n'.join(map(str, indices_list[i]))

            # Write the content to the file
            file.write(content)
