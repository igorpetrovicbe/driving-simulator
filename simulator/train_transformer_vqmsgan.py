import random

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn

import image_stuff
from custom_losses import CDWCELoss
from transformer import MultiLayerTransformer
#from dataset import DrivingSimulatorDataset
from dataset_experimental import DrivingSimulatorDataset
import cv2
import os
import numpy as np
import itertools

import wandb

from vqvae import VQVAE
from msgan import MSG_Generator

img_vocab_size = 1024

past_length = 1
exponential_base = 4
hor_size = 68
vert_size = 17
img_length = vert_size * hor_size
image_embedding_dim = 256
quantization_bins = 128

input_sequence_length = past_length * img_length + 2 * (exponential_base ** (past_length - 1))
context_sequence_length = img_length + 2
total_vocab_size = img_vocab_size + 2 * quantization_bins + 1

# Set the seed for PyTorch
seed = 42
torch.manual_seed(seed)

# Set the seed for NumPy if you are using NumPy functions
np.random.seed(seed)


def embed_indices(vqvae, indices):
    with torch.no_grad():
        quantizer = vqvae.get_quantizer()
        indices = torch.tensor(indices)
        #test_indices = indices.view(indices.shape[0], 32, 2)
        #test_indices2 = test_indices.transpose(1, 2)
        #corrected_indices = test_indices2.reshape(1, -1)
        x_tensor = torch.zeros((1, image_embedding_dim, vert_size, indices.shape[1] // vert_size))
        input_embedding = quantizer.quantize_indices(x_tensor, indices)
        input_embedding = input_embedding.squeeze(0).permute(1, 2, 0)
        input_embedding = input_embedding.view(1, input_embedding.shape[1] * vert_size, input_embedding.shape[2])
    return input_embedding


def generate_integer_sequence(model, vqvae, device, start_idx, dataset, num_frames=100, temperature=1.0):
    model.eval()  # Set the model to evaluation mode

    input_sequence = dataset.getitem_for_generation(start_idx, [], [], [])

    start_imgs, start_angles, start_spds = dataset.get_input_all(start_idx)

    context_sequence = [total_vocab_size - 1]

    generated_sequence = []

    image_indices = []
    angle_indices = []
    spd_indices = []

    step = 0
    with torch.no_grad():
        for i in range(num_frames):
            for j in range(img_length + 2):
                # Convert the seed sequence to tensor
                input_tensor = torch.tensor(input_sequence).unsqueeze(0).to(device)
                context_tensor = torch.tensor(context_sequence).unsqueeze(0).to(device)

                # Forward pass
                output = model(input_tensor, context_tensor)

                output_np = output.cpu().numpy()

                output = output[:, -1, :]

                output_np_last = output.cpu().numpy()

                # Apply temperature to the output to control randomness
                output = output.squeeze(0) / temperature

                spd_window = 11
                angle_window = 11

                if j == img_length:
                    output[:img_vocab_size + quantization_bins] = float('-inf')
                    #output[:input_tensor[0][-2].item() - angle_window // 2] = float('-inf')
                    #output[input_tensor[0][-2].item() + angle_window // 2:] = float('-inf')
                elif j == img_length + 1:
                    output[:img_vocab_size] = float('-inf')
                    output[img_vocab_size + quantization_bins:] = float('-inf')
                    #output[:input_tensor[0][-1].item() - spd_window // 2] = float('-inf')
                    #output[input_tensor[0][-1].item() + spd_window // 2:] = float('-inf')
                else:
                    output[img_vocab_size:] = float('-inf')

                output_np_last_temped = output.cpu().numpy()

                probabilities = torch.nn.functional.softmax(output, dim=0)

                probabilities_np = probabilities.cpu().numpy()

                # Sample the next index based on the probabilities
                next_index = torch.multinomial(probabilities, 1).item()

                #if j == img_length:
                    #next_index += 1
                #    next_index = 8386

                # Add the next index to the generated sequence
                generated_sequence.append(next_index)
                context_sequence.append(next_index)

                if (step + 1) % 10 == 0:
                    print(f'Step: {step}/{num_frames * (img_length + 2)}')

                step += 1

            angle_indices.append(context_sequence[-2])
            spd_indices.append(context_sequence[-1])
            image_indices = image_indices + context_sequence[-2 - img_length:-2]
            input_sequence = dataset.getitem_for_generation(start_idx, image_indices, angle_indices, spd_indices)
            context_sequence = [total_vocab_size - 1]

    return start_imgs + image_indices, start_angles + angle_indices, start_spds + spd_indices


def select_img_elements_old(tensor):
    selected_tensors = []
    total_length = img_length + 2
    for i in range(tensor.shape[0] // total_length):
        selected_tensors.append(tensor[i*total_length:i*total_length + img_length, :img_vocab_size])

    return torch.cat(selected_tensors, dim=0)


def select_img_elements(tensor):
    total_length = img_length + 2
    num_chunks = tensor.shape[0] // total_length

    selected_tensors = torch.stack([
        tensor[i * total_length:i * total_length + img_length, :img_vocab_size]
        for i in range(num_chunks)
    ])

    return selected_tensors.reshape(-1, img_vocab_size)


# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

load_mode = False
force_new_lr = False

if load_mode:
    loaded_state = torch.load('saved_state_iteration_134999_v4_small_undertrained.pth')

# Initialize model.
use_ema = True
vqvae_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 3,
    "num_residual_layers": 2,
    "num_residual_hiddens": 128,
    "embedding_dim": 256,
    "num_embeddings": img_vocab_size,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}

vqvae = VQVAE(**vqvae_args).to(device)

micro_batch_size = 8
accumulated_batch_size = 256
gradient_accumulation_steps = accumulated_batch_size // micro_batch_size
d_model = 768
ff_size = 4 * d_model
output_size = img_vocab_size + 2 * quantization_bins
num_layers = 12
num_heads = int(d_model / 64)
learning_rate = 5e-4
learning_rate_end = learning_rate * 0.1
num_epochs = 100000
weight_decay = 0.1

# Set the warm-up parameters
use_warmup = True
warmup_iters = 1000 * gradient_accumulation_steps
warmup_init_lr = learning_rate * 0.1  # 1e-5

# Create dataset and data loader
current_directory = os.getcwd()
dataset_path = os.path.join(current_directory, 'dataset4')
print(dataset_path)
dataset = DrivingSimulatorDataset(root_dir=dataset_path, device=device, img_length=img_length,
                                  img_vocab_size=img_vocab_size, exponential_base=exponential_base,
                                  past_length=past_length, quantization_bins=quantization_bins)
data_loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True)

#for batch_idx in enumerate(data_loader):
#    print(batch_idx)

model = MultiLayerTransformer(d_model, total_vocab_size, ff_size, num_layers, num_heads, input_sequence_length,
                              context_sequence_length)
if load_mode:
    model.load_state_dict(loaded_state['model_state_dict'])

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

# Count the number of parameters in VQ VAE
num_params = sum(p.numel() for p in vqvae.parameters())
print(f"Number of parameters in the VQ VAE: {num_params}")

vqvae.load_state_dict(torch.load('vqvae_v4.pth'))

msgan = MSG_Generator().to(device)
loaded_generator_state = torch.load('msgan_v4.pth')
msgan.load_state_dict(loaded_generator_state)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
criterion_other = nn.CrossEntropyLoss()
#criterion_cdwce = CDWCELoss(alpha=5)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-6)
if load_mode:
    optimizer.load_state_dict(loaded_state['optimizer_state_dict'])

max_steps = total_params * 20 / (accumulated_batch_size * context_sequence_length)


# Add a cosine annealing learning rate scheduler
max_gradient_norm = 1.0
print(int(max_steps - warmup_iters // gradient_accumulation_steps))
scheduler = CosineAnnealingLR(optimizer, T_max=int(max_steps - warmup_iters // gradient_accumulation_steps), eta_min=learning_rate_end)
if load_mode:
    scheduler.load_state_dict(loaded_state['scheduler_state_dict'])

# Set the value of K for plotting every K steps
generate_frequency = 100
save_every_n_iterations = 1000
losses = []  # To store the training losses
running_loss = 0.0  # To accumulate the loss over K steps

iteration = 0
start_batch_idx = 0
if load_mode:
    iteration = loaded_state['iteration'] + 1
    start_batch_idx = loaded_state['batch_idx'] + 1
    running_loss = 0.0

    # Set the learning rate in the optimizer based on the current iteration
    if iteration < warmup_iters:
        lr = warmup_init_lr + (learning_rate - warmup_init_lr) * (iteration / warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif force_new_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate


    print('Preskip point')
    skipped_batches = itertools.islice(data_loader, start_batch_idx, None)
    print('Skipped1')
    data_loader.__iter__ = skipped_batches.__iter__
    print('Skipped2')

# Initialize other variables needed for saving
saved_state = {
    'iteration': iteration,
    'batch_idx': start_batch_idx,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'running_loss': running_loss,
}

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="driving-simulator",

    name=f"Transformer Full {d_model}d {num_layers}l {num_heads}heads",

    # track hyperparameters and run metadata
    config={
        "learning_rate_start": learning_rate,
        "learning_rate_end": learning_rate_end,
        "context_length": context_sequence_length,
        "memory_length": input_sequence_length,
        "ff_size": ff_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": accumulated_batch_size,
        "warmup_iters": warmup_iters,
        "warmup_init_lr": warmup_init_lr,
        "weight_decay": weight_decay,
        "dataset": "v4",
    },
)
#    resume="must",
#    id="whmjtdf4"
total_loss = 0

if __name__ == "__main__":
    for epoch in range(num_epochs):
        for step, (inputs, context, targets) in enumerate(data_loader):
            model.train()  # Set the model to training mode
            # Move inputs and targets to GPU
            inputs, context, targets = inputs.to(device), context.to(device), targets.to(device)

            # Update learning rate during warm-up phase
            if use_warmup and iteration < warmup_iters:
                lr = warmup_init_lr + (learning_rate - warmup_init_lr) * (iteration / warmup_iters)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if iteration > max_steps * gradient_accumulation_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate_end
            # Forward pass
            outputs = model(inputs, context)
            #log_probs = F.log_softmax(outputs, dim=-

            reshaped_outputs = outputs.reshape(-1, total_vocab_size - 1)
            reshaped_targets = targets.reshape(-1, total_vocab_size - 1)

            #np_reshaped_outputs = reshaped_outputs.detach().cpu().numpy()
            #np_reshaped_targets = reshaped_targets.detach().cpu().numpy()

            img_outputs = select_img_elements(reshaped_outputs)
            img_targets = select_img_elements(reshaped_targets)

            #np_image_outputs = img_outputs.detach().cpu().numpy()
            #np_image_targets = img_targets.detach().cpu().numpy()

            angle_outputs = reshaped_outputs[img_length::img_length+2, img_vocab_size + quantization_bins:]
            angle_targets = reshaped_targets[img_length::img_length+2, img_vocab_size + quantization_bins:]

            #np_angle_outputs = angle_outputs.detach().cpu().numpy()
            #np_angle_targets = angle_targets.detach().cpu().numpy()

            spd_outputs = reshaped_outputs[img_length+1::img_length+2, img_vocab_size:img_vocab_size + quantization_bins]
            spd_targets = reshaped_targets[img_length+1::img_length+2, img_vocab_size:img_vocab_size + quantization_bins]

            #np_spd_outputs = spd_outputs.detach().cpu().numpy()
            #np_spd_targets = spd_targets.detach().cpu().numpy()

            # Compute the loss
            loss_img = criterion(img_outputs, img_targets)
            loss_angle = criterion_other(angle_outputs, angle_targets) * 0
            loss_spd = criterion_other(spd_outputs, spd_targets) * 0
            loss = loss_img + loss_angle + loss_spd
            #loss = criterion(log_probs, targets)

            # Check for NaN or Inf loss
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                print(f"NaN or Inf loss encountered at epoch {epoch + 1}, step {step + 1}. Skipping update step.")
                continue

            # Backward pass and optimization
            loss.backward()

            total_loss += loss.item()

            if (iteration + 1) % gradient_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

                # Print loss every K steps
                average_loss = total_loss / gradient_accumulation_steps
                current_lr = scheduler.get_lr()[0]  # Get the current learning rate
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(data_loader)}], Average Loss: {average_loss}'
                    f', Learning Rate: {current_lr}')

                # Store the average loss for plotting
                losses.append(average_loss)

                wandb.log({"loss": average_loss})

                optimizer.step()
                optimizer.zero_grad()

                total_loss = 0.0

                # Update the learning rate
                scheduler.step()

            print(f'Iteration: {iteration}/{max_steps * gradient_accumulation_steps}')

            if (iteration + 1) % (generate_frequency * gradient_accumulation_steps) == 0 or iteration + 1 == max_steps:
                model.eval()
                with torch.no_grad():
                    # Choose a random seed sequence from the dataset
                    random_idx = random.randint(0, len(dataset) - 1)
                    seed_sequence = dataset[random_idx][0]  # Take the input sequence from the dataset

                    seed_list = seed_sequence.tolist()
                    video_length = 10
                    # Generate text using the trained RNN
                    image_indices, angle_indices, spd_indices = generate_integer_sequence(model, vqvae, device,
                                                                                          random_idx, dataset,
                                                                                          num_frames=video_length,
                                                                                          temperature=1.0)

                    for k in range(len(image_indices) // img_length):
                        current_image_indices = image_indices[k * img_length: (k+1) * img_length]
                        total_length = len(current_image_indices)
                        print(f'max: {max(current_image_indices)}')
                        print(f'min: {min(current_image_indices)}')

                        current_image_indices = torch.tensor(current_image_indices).to(device)

                        test_indices = current_image_indices.view(-1, total_length // vert_size, vert_size)
                        test_indices2 = test_indices.transpose(1, 2)
                        corrected_indices = test_indices2.reshape(-1)

                        quantizer = vqvae.get_quantizer()
                        x_tensor = torch.zeros((1, vqvae_args['embedding_dim'], vert_size, total_length // vert_size))  # 2 x 128
                        quantized_x = quantizer.quantize_indices(x_tensor, corrected_indices)

                        gen_imgs = msgan(quantized_x)

                        outputs = gen_imgs[-2].squeeze(0)

                        #outputs = vqvae.decode(quantized_x).to('cpu')
                        #outputs = outputs.squeeze(0)

                        image_np = outputs.cpu().detach().numpy()
                        image_np = np.transpose(image_np, (1, 2, 0))

                        # Normalize the image if necessary
                        image_np = (image_np - image_np.min()) / (
                                    image_np.max() - image_np.min())  # Normalize to [0, 1]
                        image_np = (image_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                        if k > 0:
                            angle = dataset.dequantize_angle(angle_indices[k-1])
                            speed = dataset.dequantize_spd(spd_indices[k-1])

                            # Load the steering wheel sprite image
                            steering_wheel_img = cv2.imread('assets/steering-wheel.png', cv2.IMREAD_UNCHANGED)
                            steering_wheel_img = cv2.resize(steering_wheel_img, (25, 25))
                            if steering_wheel_img is None:
                                print("Error: Could not load steering wheel image.")

                            # Get the dimensions of the steering wheel image
                            wheel_h, wheel_w, _ = steering_wheel_img.shape

                            # Rotate the steering wheel image based on the steering angle
                            rotated_wheel = image_stuff.rotate_image(steering_wheel_img, angle)

                            # Determine the position to place the steering wheel image
                            wheel_x = 25
                            wheel_y = int(image_np.shape[0] - wheel_h - 25)

                            # Overlay the rotated steering wheel image onto the frame
                            image_np = image_stuff.overlay_rgba_on_rgb(image_np, rotated_wheel, wheel_x, wheel_y)

                            cv2.putText(image_np, f"Speed: {speed:.2f} m/s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                                        (255, 255, 255), 1)

                        cv2.imwrite(f'generated_images/{(iteration + 1) // gradient_accumulation_steps}_{k}.png', image_np)

                        # Print the generated indices
                        print(f"Generated {len(current_image_indices)} image indices:")
                        print(current_image_indices)

            if (iteration + 1) % save_every_n_iterations == 0 or iteration + 1 == max_steps:
                saved_state['iteration'] = iteration
                saved_state['batch_idx'] = step
                saved_state['model_state_dict'] = model.state_dict()
                saved_state['optimizer_state_dict'] = optimizer.state_dict()
                saved_state['scheduler_state_dict'] = scheduler.state_dict()
                saved_state['running_loss'] = running_loss

                torch.save(saved_state, f'saved_state_iteration_{iteration}.pth')

            iteration += 1


    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
