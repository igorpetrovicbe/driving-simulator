import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from vqvae import VQVAE

# Assuming you have a custom dataset loader; adjust as needed
from image_dataset import ImageDataset

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Generator
class MSG_Generator(nn.Module):
    def __init__(self, img_channels=3, input_h_dim=256, start_h_dim=512, layers_per_block=5):
        super(MSG_Generator, self).__init__()
        self.start_h_dim = start_h_dim

        self.init_size = (25, 34)  # Initial size before upsampling

        self.pre_conv = nn.Conv2d(in_channels=input_h_dim, out_channels=start_h_dim, kernel_size=3, stride=1, padding=1)

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(start_h_dim),
            nn.Upsample(scale_factor=2)
        )

        for i in range(layers_per_block - 1):
            self.block1.add_module(f'conv_{i+1}', nn.Conv2d(start_h_dim, start_h_dim, kernel_size=3, stride=1, padding=1))
            self.block1.add_module(f'bn_{i+1}',nn.BatchNorm2d(start_h_dim, 0.8))
            self.block1.add_module(f'lrelu_{i+1}',nn.LeakyReLU(0.2, inplace=True))
        self.block1.add_module(f'conv_{layers_per_block}', nn.Conv2d(start_h_dim, start_h_dim, kernel_size=3, stride=1, padding=1))
        self.block1.add_module(f'bn_{layers_per_block}', nn.BatchNorm2d(start_h_dim, 0.8))
        self.block1.add_module(f'lrelu_{layers_per_block}', nn.LeakyReLU(0.2, inplace=True))

        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )
        for i in range(layers_per_block - 1):
            self.block2.add_module(f'conv_{i+1}', nn.Conv2d(start_h_dim, start_h_dim, kernel_size=3, stride=1, padding=1))
            self.block2.add_module(f'bn_{i+1}',nn.BatchNorm2d(start_h_dim, 0.8))
            self.block2.add_module(f'lrelu_{i+1}',nn.LeakyReLU(0.2, inplace=True))
        self.block2.add_module(f'conv_{layers_per_block}', nn.Conv2d(start_h_dim, start_h_dim // 2, kernel_size=3, stride=1, padding=1))
        self.block2.add_module(f'bn_{layers_per_block}', nn.BatchNorm2d(start_h_dim // 2, 0.8))
        self.block2.add_module(f'lrelu_{layers_per_block}', nn.LeakyReLU(0.2, inplace=True))

        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )
        for i in range(layers_per_block - 1):
            self.block3.add_module(f'conv_{i+1}', nn.Conv2d(start_h_dim // 2, start_h_dim // 2, kernel_size=3, stride=1, padding=1))
            self.block3.add_module(f'bn_{i+1}',nn.BatchNorm2d(start_h_dim // 2, 0.8))
            self.block3.add_module(f'lrelu_{i+1}',nn.LeakyReLU(0.2, inplace=True))
        self.block3.add_module(f'conv_{layers_per_block}', nn.Conv2d(start_h_dim // 2, start_h_dim // 4, kernel_size=3, stride=1, padding=1))
        self.block3.add_module(f'bn_{layers_per_block}', nn.BatchNorm2d(start_h_dim // 4, 0.8))
        self.block3.add_module(f'lrelu_{layers_per_block}', nn.LeakyReLU(0.2, inplace=True))

        self.final_conv1 = nn.Conv2d(start_h_dim, img_channels, kernel_size=3, stride=1, padding=1)
        self.final_conv2 = nn.Conv2d(start_h_dim // 2, img_channels, kernel_size=3, stride=1, padding=1)
        self.final_conv3 = nn.Conv2d(start_h_dim // 4, img_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        #out = self.l1(z)
        #out = out.view(out.shape[0], self.start_h_dim, self.init_size[0], self.init_size[1])

        out = self.pre_conv(z)

        out1 = self.block1(out)
        img1 = self.tanh(self.final_conv1(out1))

        out2 = self.block2(out1)
        img2 = self.tanh(self.final_conv2(out2))

        out3 = self.block3(out2)
        img3 = self.tanh(self.final_conv3(out3))

        return [img1, img2, img3]


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, start_h_dim=32, n_blocks=6, layers_per_block=3):
        super(Discriminator, self).__init__()

        self.layers_per_block = layers_per_block
        self.discriminator_blocks = nn.ModuleList()
        in_channels = img_channels
        out_channels = start_h_dim

        for i in range(n_blocks):
            self.discriminator_blocks.append(self._make_disc_block(in_channels, out_channels, layers_per_block, first_block=(i == 0)))
            in_channels = out_channels
            out_channels *= 2

        self.fc = nn.Linear(out_channels // 2 * 1 * 4, 1)  # Assuming the smallest feature map is 2x4
        self.sigmoid = nn.Sigmoid()

    def _make_disc_block(self, in_channels, out_channels, layers_per_block, first_block=False):
        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        if not first_block:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        #if not first_block:
        #    layers.append(nn.Dropout2d(0.25))

        for i in range(layers_per_block - 2):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            #layers.append(nn.Dropout2d(0.25))

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        #layers.append(nn.Dropout2d(0.25))
        return nn.Sequential(*layers)

    def forward(self, img):
        out = img
        for block in self.discriminator_blocks:
            residual = out
            for layer in block[:-3]:  # Apply all layers except the last 3 (before downsampling)
                out = layer(out)
            out += residual  # Add the residual connection
            for layer in block[-3:]:  # Apply the remaining layers (including downsampling)
                out = layer(out)
        out = out.view(out.shape[0], -1)
        validity = self.sigmoid(self.fc(out))
        return validity


# Multi-scale loss function
class MultiScaleLoss(nn.Module):
    def __init__(self, loss_weights=[1, 1, 1, 1]):
        super(MultiScaleLoss, self).__init__()
        self.loss_weights = loss_weights
        self.bce_loss = nn.BCELoss()

    def forward(self, disc_outputs, labels):
        total_loss = 0
        for i, (output, weight) in enumerate(zip(disc_outputs, self.loss_weights)):
            total_loss += weight * self.bce_loss(output, labels)
        return total_loss


# Loss functions
adversarial_loss = MultiScaleLoss()
mse_loss = nn.MSELoss()

# Create output image directory
os.makedirs("images", exist_ok=True)

# Training parameters
epochs = 2000
full_batch_size = 16
micro_batch_size = 4
gradient_accumulation_steps = full_batch_size // micro_batch_size

z_dim = 100
#dataset_path = "driving_images.h5"
#dataset_path = 'H:\\PycharmProjects\\self_driving_reader\\driving_images4.h5'
dataset_path = "D:\\datasets\\driving_images4.h5"

# Data loading
transform = transforms.Compose([
    transforms.Resize((400, 544)),
    transforms.ToTensor()
])

dataset = ImageDataset(h5_file_path=dataset_path, transform=None)
dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True, num_workers=8)
iterator = 0

# Initialize model.
use_ema = True
vqvae_args = {
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

warmup_iterations = 0  # 100000 / micro_batch_size

init_lambda_adversarial = 0.001
lambda_reconstruction = 1.0

export_image_every = 10
save_every = 1000

# Training Loop
if __name__ == "__main__":
    # Initialize models
    generator = MSG_Generator().to(device)
    discriminators = nn.ModuleList([
        Discriminator(n_blocks=5).to(device),  # For 34x136
        Discriminator(n_blocks=6).to(device),  # For 68x272
        Discriminator(n_blocks=7).to(device),  # For 136x544
    ])

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=2e-5, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(sum([list(d.parameters()) for d in discriminators], []), lr=2e-5, betas=(0.5, 0.999))

    # Count the number of parameters
    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Number of parameters in the generator: {num_params}")

    # Count the number of parameters
    num_params = sum(p.numel() for p in discriminators[2].parameters())
    print(f"Number of parameters in the discriminator: {num_params}")

    vqvae = VQVAE(**vqvae_args).to(device)
    # vqvae = torch.load('vqvae_7999.pth').to(device)
    vqvae.load_state_dict(torch.load('vqvae_v4_2999.pth'))

    vqvae.eval()
    generator.train()
    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            # Adversarial ground truths
            imgs, _ = imgs
            valid = torch.ones(imgs.size(0), 1, device=device, dtype=torch.float32)
            fake = torch.zeros(imgs.size(0), 1, device=device, dtype=torch.float32)

            # Configure input
            unnormalized_imgs = imgs.to(device)
            real_imgs = transforms.Normalize([0.5], [0.5])(unnormalized_imgs / 255)

            # Train Generator
            optimizer_G.zero_grad()
            with torch.no_grad():
                (quantized_imgs, _, _, _) = vqvae.quantize(unnormalized_imgs)
                quantized_imgs = quantized_imgs.detach() / 255
            gen_imgs = generator(quantized_imgs)

            g_loss = 0
            j = 1
            for scale_img, discriminator in zip(gen_imgs, discriminators):
                validity = discriminator(scale_img)
                L_GAN = adversarial_loss([validity], valid)

                total_scales = len(gen_imgs)
                ground_truth_img = torch.nn.functional.avg_pool2d(real_imgs, kernel_size=2 ** (total_scales - j),
                                                                  stride=2 ** (total_scales - j), padding=0)
                L_rec = mse_loss(scale_img, ground_truth_img)

                if j == 1:
                    last_layer = generator.final_conv1
                elif j == 2:
                    last_layer = generator.final_conv2
                else:
                    last_layer = generator.final_conv3

                # Compute gradients of the reconstruction loss with respect to the last layer's parameters
                grad_L_rec = torch.autograd.grad(L_rec, last_layer.parameters(), retain_graph=True, create_graph=True)

                # Compute gradients of the GAN loss with respect to the last layer's parameters
                grad_L_GAN = torch.autograd.grad(L_GAN, last_layer.parameters(), retain_graph=True, create_graph=True)

                # Compute the norms of the gradients
                grad_norm_L_rec = sum(g.norm() for g in grad_L_rec)
                grad_norm_L_GAN = sum(g.norm() for g in grad_L_GAN)

                # Compute the adaptive weight lambda
                delta = 1e-6  # Small constant to prevent division by zero

                #if iterator > warmup_iterations:
                #    lambda_adaptive = grad_norm_L_rec / (grad_norm_L_GAN + delta)
                #    print(lambda_adaptive)
                #else:
                #    lambda_adaptive = init_lambda_adversarial

                g_loss += L_GAN * init_lambda_adversarial
                g_loss += L_rec

                j += 1

            g_loss.backward()
            if (iterator + 1) % gradient_accumulation_steps == 0:
                optimizer_G.step()

            # Train Discriminators
            optimizer_D.zero_grad()

            d_loss = 0
            for scale_img, discriminator in zip(gen_imgs, discriminators):
                real_img_scaled = nn.functional.interpolate(real_imgs, size=scale_img.shape[2:])
                real_validity = discriminator(real_img_scaled)
                fake_validity = discriminator(scale_img.detach())

                real_loss = adversarial_loss([real_validity], valid)
                fake_loss = adversarial_loss([fake_validity], fake)
                d_loss += (real_loss + fake_loss) / 2

            d_loss.backward()
            if (iterator + 1) % gradient_accumulation_steps == 0:
                optimizer_D.step()

            print(
                f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            if (iterator + 1) % (save_every * gradient_accumulation_steps) == 0:
                torch.save(generator.state_dict(), 'generator.pth')
                for j, discriminator in enumerate(discriminators):
                    torch.save(discriminator.state_dict(), f'discriminator_{j}.pth')

            if (iterator + 1) % (export_image_every * gradient_accumulation_steps) == 0:
                # Save images at sample intervals
                batches_done = epoch * len(dataloader) + i
                # Combine images from all scales into a single grid
                all_scale_imgs = []
                for j, img in enumerate(gen_imgs):
                    # Select a subset of images if there are too many
                    num_images = min(8, img.size(0))  # Limit to 16 images per scale
                    #num_images = img.size(0)
                    scale_imgs = img[:num_images]

                    # Scale smaller images to match the largest scale dimensions using nearest neighbor
                    if j < len(gen_imgs) - 1:
                        scale_imgs = nn.functional.interpolate(scale_imgs, size=gen_imgs[-1].shape[2:],
                                                               mode='nearest')

                    all_scale_imgs.append(scale_imgs)

                # Concatenate all scales
                all_scale_imgs = torch.cat(all_scale_imgs, dim=0)

                # Create and save the grid
                grid = make_grid(all_scale_imgs, nrow=num_images, normalize=True, scale_each=True)
                save_image(grid, f"images/{epoch}_{iterator // gradient_accumulation_steps}_all_scales.png")

            iterator += 1