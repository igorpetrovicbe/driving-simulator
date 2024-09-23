# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.
import os
import random

import numpy as np
import torch
import torchaudio

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from image_dataset import ImageDataset
from vqvae import VQVAE, Discriminator
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
import torchvision.transforms.functional as TF

torch.set_printoptions(linewidth=160)


def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array > 255] = 255
    imgs_array[imgs_array < 0] = 0
    (batch_size, img_vsize, img_hsize) = img_tensors.shape[:3]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * img_vsize, ncols * img_hsize, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_vsize
        row_end = row_start + img_vsize
        col_start = col_idx * img_hsize
        col_end = col_start + img_hsize
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.png")


def calculate_variance(dataset, sample_size=1000):
    # Ensure sample size is not larger than the dataset
    sample_size = min(sample_size, len(dataset))

    # Create a random sample of indices
    indices = random.sample(range(len(dataset)), sample_size)

    # Collect pixel values from the sampled images
    all_pixels = []
    for idx in tqdm(indices, desc="Calculating variance", dynamic_ncols=True):
        img, _ = dataset[idx]
        if isinstance(img, torch.Tensor):
            img_array = img.numpy().flatten()
        else:
            img_array = np.array(img).flatten()
        all_pixels.extend(img_array)

    # Calculate overall variance
    variance = np.var(all_pixels)
    return variance


def sobel_filter(x):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        x.device)

    grad_x = F.conv2d(x, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    grad_y = F.conv2d(x, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

    return torch.sqrt(grad_x ** 2 + grad_y ** 2)


def rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    max_rgb, _ = torch.max(rgb, dim=1)
    min_rgb, _ = torch.min(rgb, dim=1)
    diff = max_rgb - min_rgb

    h = torch.zeros_like(max_rgb)
    s = torch.zeros_like(max_rgb)
    v = max_rgb

    # Hue calculation
    mask = (max_rgb != min_rgb)
    mask_r = mask & (r == max_rgb)
    mask_g = mask & (g == max_rgb)
    mask_b = mask & (b == max_rgb)

    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

    # Saturation calculation
    s[mask] = diff[mask] / max_rgb[mask]

    return torch.stack([h, s, v], dim=1) / 360.0  # Normalize to [0, 1]


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
    #vqvae.load_state_dict(torch.load('vqvae.pth')['model_state_dict'])

    discriminator = Discriminator(model_args['in_channels'], model_args['num_hiddens']).to(device)

    # Count the number of parameters
    num_discriminator_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Number of parameters in the discriminator: {num_discriminator_params}")

    # Initialize dataset.
    full_batch_size = 32
    micro_batch_size = 32
    gradient_accumulation_steps = full_batch_size // micro_batch_size
    workers = 1  # 4

    #dataset_path = 'H:\\PycharmProjects\\VQGAN-pytorch-main\\driving_images4.h5'
    dataset_path = "D:\\datasets\\driving_images4.h5"

    train_dataset = ImageDataset(h5_file_path=dataset_path, transform=None)

    # Use the function
    sample_size = 1000  # Adjust this value based on your needs
    train_data_variance = calculate_variance(train_dataset, sample_size)

    print(f"Estimated overall variance (from {sample_size} samples): {train_data_variance}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=micro_batch_size,
        shuffle=True
    )

    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation
    # Learning".
    beta = 0.25

    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    discriminator_params = [params for params in discriminator.parameters()]

    lr = 1e-4

    #optimizer = optim.Adam(train_params, lr=lr, betas=(0.5, 0.999))
    optimizer = optim.Adam(train_params, lr=lr)

    criterion = nn.MSELoss()
    criterion_discriminator = nn.BCELoss()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="self-driving-vqvae",

        name=f"{model_args['num_residual_hiddens']}h {model_args['num_downsampling_layers']}down "
             f"{full_batch_size}b {0}lmbd",

        # track hyperparameters and run metadata
        config={
            "learning_rate_start": lr,
            "hidden_size": model_args['num_hiddens'],
            "num_downsampling_layers": model_args['num_downsampling_layers'],
            "num_residual_layers": model_args['num_residual_layers'],
            "batch_size": full_batch_size,
            "dataset": "v4",
        }
    )

    # Train model.
    epochs = 1000
    eval_every = 1
    generate_every = 100
    save_every = 1000
    best_train_loss = float("inf")
    iteration = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_recon_error = 0
        total_discriminator_loss = 0
        total_vq_discriminator_loss = 0
        total_accuracy = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs, _ = train_tensors
            imgs = imgs.to(device)

            hsv_imgs = rgb_to_hsv(imgs)
            edge_map = sobel_filter(hsv_imgs)
            edge_map, _ = torch.max(edge_map, dim=1, keepdim=True)
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())

            edge_weight_factor = 5.0

            out = model(imgs)

            #mse_loss = F.mse_loss(out["x_recon"], imgs, reduction='none')
            #weighted_mse_loss = mse_loss * (1 + edge_weight_factor * edge_map)
            #recon_error = weighted_mse_loss.mean() / train_data_variance

            recon_error = criterion(out["x_recon"], imgs) / train_data_variance

            total_recon_error += recon_error.item()
            vq_vae_loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                vq_vae_loss += out["dictionary_loss"]

            if (iteration + 1) % gradient_accumulation_steps == 0:
                wandb.log({"loss": recon_error.item()})

            total_train_loss += vq_vae_loss.item()

            total_train_loss += vq_vae_loss.item()

            vq_vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (iteration + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

            n_train += 1

            if ((iteration + 1) % eval_every) == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                print(f"iteration: {iteration}")
                total_train_loss /= n_train
                if total_train_loss < best_train_loss:
                    best_train_loss = total_train_loss

                print(f"total_train_loss: {total_train_loss}")
                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {total_recon_error / n_train}\n")

                total_train_loss = 0
                total_recon_error = 0
                n_train = 0

            if (iteration + 1) % save_every == 0:
                torch.save(model.state_dict(), f'vqvae_{iteration}.pth')

            if (iteration + 1) % generate_every == 0:
                # Generate and save reconstructions.
                model.eval()
                valid_dataset = train_dataset
                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=micro_batch_size,
                    shuffle=True
                )

                with torch.no_grad():
                    for valid_tensors in valid_loader:
                        break

                    valid_img_tensor, _ = valid_tensors

                    save_img_tensors_as_grid(valid_img_tensor, 4, f"vqvae_images/{iteration}_true")
                    save_img_tensors_as_grid(model(valid_img_tensor.to(device))["x_recon"], 4, f"vqvae_images/{iteration}_recon")

            iteration += 1
