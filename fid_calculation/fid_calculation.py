import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
import os
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from image_dataset_fid import ImageDataset
import torch.nn as nn


# Custom Dataset for Generated Images that end with '0'
class GeneratedImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        number = '6'
        self.folder_path = folder_path
        # Correctly filter only images whose filenames end with '0'
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                            f.endswith(f'{number}.jpg') or f.endswith(f'{number}.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Open image as RGB

        if self.transform:
            image = self.transform(image)

        return image


# Transform for generated and real images with zoom (resize and center crop)
transform_gen = transforms.Compose([
    transforms.Resize(299),  # Resize smaller dimension to 299, maintaining aspect ratio
    transforms.CenterCrop(299),  # Center crop the image to 299x299
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for Inception v3
])

# Create the Generated Image Dataset and DataLoader
generated_folder_path = 'H:\\PycharmProjects\\driving_simulator\\test_measurement_images'
gen_dataset = GeneratedImageDataset(generated_folder_path, transform=transform_gen)
gen_loader = DataLoader(gen_dataset, batch_size=10, shuffle=True)

# Real Dataset - Applying the same transformation to real images
#real_folder_path = "D:\\datasets\\driving_images4.h5"
real_folder_path = "H:\\PycharmProjects\\self_driving_reader\\driving_images_test.h5"
real_dataset = ImageDataset(real_folder_path, transform=transform_gen)  # Apply the same transformation here
real_loader = DataLoader(real_dataset, batch_size=10, shuffle=True)


import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(real_features, generated_features):
    # Calculate mean and covariance for both feature sets
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)

    cov_real = np.cov(real_features, rowvar=False)
    cov_generated = np.cov(generated_features, rowvar=False)

    # Calculate squared difference between means
    diff = mu_real - mu_generated
    mean_diff = np.sum(diff ** 2)

    # Add a small epsilon to the diagonal of the covariance matrices for numerical stability
    epsilon = 1e-6
    cov_real += np.eye(cov_real.shape[0]) * epsilon
    cov_generated += np.eye(cov_generated.shape[0]) * epsilon

    # Calculate matrix square root of the product of covariances using scipy
    cov_mean = sqrtm(cov_real @ cov_generated)

    # Check if the resulting cov_mean is complex
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    # Calculate the final FID score
    fid = mean_diff + np.trace(cov_real + cov_generated - 2 * cov_mean)

    return fid



class InceptionFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(InceptionFeatureExtractor, self).__init__()
        # Store the layers, excluding the final classification layer and auxiliary layer
        self.layers = []
        for name, layer in model.named_children():
            if name not in ['AuxLogits', 'fc']:  # Exclude auxiliary and final layers
                self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)  # Convert to ModuleList for proper handling

    def forward(self, x):
        # Pass the input through each layer one by one
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(f"After layer {i} ({layer.__class__.__name__}): shape {x.shape}")
        return x.view(x.shape[0], x.shape[1])


# Load Inception v3 model for feature extraction and move it to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load Inception v3 model
inception = inception_v3(pretrained=True, transform_input=False).to(device).eval()

inception_feature_extractor = InceptionFeatureExtractor(inception)


# Function to extract features using Inception v3
def extract_features(loader, max_samples):
    features = []
    for i, batch in enumerate(loader):
        if (i + 1) % 1 == 0:
            print(i * loader.batch_size)
        if i >= max_samples // loader.batch_size:
            break
        with torch.no_grad():
            # Move the batch to GPU
            batch = batch.to(device)
            # Pass batch through Inception v3, get features from the pool3 layer (2048-dim)
            output = inception_feature_extractor(batch).detach().cpu().numpy()  # Move output back to CPU
        features.append(output)
    return np.concatenate(features, axis=0)


# Debugging: Visualize a batch of transformed images
def denormalize_and_show(tensor, mean, std):
    """ Denormalize the image tensor and plot it. """
    # Undo normalization
    tensor = tensor.clone()  # Clone to avoid modifying original tensor
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Scale by std and add mean

    # Convert the tensor to a numpy array for plotting
    np_image = tensor.permute(1, 2, 0).cpu().numpy()

    # Clip any values out of range [0, 1]
    np_image = np.clip(np_image, 0, 1)

    return np_image


def show_transformed_images(loader):
    data_iter = iter(loader)
    images = next(data_iter)  # Get a batch of images

    # Denormalize and convert images to display format
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Plot the first few images from the batch
    plt.figure(figsize=(12, 8))
    for i in range(6):  # Display first 6 images
        image = images[i]
        np_image = denormalize_and_show(image, mean, std)

        plt.subplot(2, 3, i + 1)
        plt.imshow(np_image)
        plt.axis('off')

    plt.show()


# Sampling with replacement
def sample_from_dataset(dataset, num_samples):
    indices = np.random.choice(len(dataset), size=num_samples, replace=True)
    subset = Subset(dataset, indices)
    return subset


# Show the images after transformation
show_transformed_images(gen_loader)  # Generated images
show_transformed_images(real_loader)  # Real images


load_mode = True
# Extract features from real and generated images
if not load_mode:
    real_features = extract_features(real_loader, 1000)
    np.save('real_features.npy', real_features)
else:
    real_features = np.load('real_features.npy')

fid_scores = []
num_samples = len(gen_dataset)

# Perform 1000 bootstrap iterations
for i in range(100):
    print(i)
    # Sample from the dataset
    sample_gen_dataset = sample_from_dataset(gen_dataset, num_samples)
    sample_gen_loader = DataLoader(dataset=sample_gen_dataset, batch_size=10)

    # Extract features from the sampled generated images
    generated_features = extract_features(sample_gen_loader, 99999)

    # Calculate FID for the current sample
    fid_score = calculate_fid(real_features, generated_features)
    print(f"Current FID: {fid_score}")
    fid_scores.append(fid_score)

# Convert FID scores to a NumPy array for easier manipulation
fid_scores = np.array(fid_scores)

# Calculate the average FID
average_fid = np.mean(fid_scores)

# Calculate the 95% confidence interval (2.5th and 97.5th percentiles)
lower_bound = np.percentile(fid_scores, 2.5)
upper_bound = np.percentile(fid_scores, 97.5)

# Print the results
print(f"Average FID: {average_fid}")
print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")