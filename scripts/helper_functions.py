# helper_functions.py
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import os
import matplotlib.pyplot as plt
from image_loader_dataset import ImageLoaderDataset


def get_image_mean_std(dataset_path, device='cuda', batch_size=512):
    """
    Compute the mean and standard deviation of images in the given dataset path.
    Return a tuple containing mean and standard deviation of dataset.
    """

    # Define image transforms (Resize images and convert them to tensors)
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Get image paths and labels from ImageFolder
    img_folder = ImageFolder(root=dataset_path)
    img_paths = [item[0] for item in img_folder.imgs]
    labels = [item[1] for item in img_folder.imgs]

    # Create dataset with transforms
    image_dataset = ImageLoaderDataset(img_paths=img_paths, label_list=labels, transform=image_transforms)

    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize variables
    mean = 0.
    std = 0.
    sample_count = 0.

    # Loop through dataset to compute mean and standard deviation
    for data, _ in tqdm(data_loader):
        data = data.to(device)  
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0).cpu()  
        std += data.std(2).sum(0).cpu()  
        sample_count += batch_samples

    mean /= sample_count
    std /= sample_count

    # Return mean and standard deviation of the images
    return mean, std


def plot_image_dimensions(img_dir, heading='Image Dimensions', save_as=None, alpha=0.1):
    """
    Plot the dimensions of images in a directory using a scatter plot.
    """
    
    # Getting all images in the directory
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpeg', 'jpg', 'png', 'webp'))]

    widths = []
    heights = []

    # Loop through each image and extract its dimensions
    for img_file in img_files:
        with Image.open(os.path.join(img_dir, img_file)) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)

    # Plot the dimensions of the images in a scatter plot
    plt.scatter(widths, heights, alpha=alpha)
    plt.title(heading)
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()

def unnormalize(img, mean, std):
    """
    Unnormalize an image tensor
    """
    img = img.clone()  # Ensure original image tensor isn't modified
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def show_img(dataloader, class_names, mean, std, num_images=24):
    """Display a grid of images from a dataloader with their labels"""

    # Get batch of images with labels from dataloader
    images, labels = next(iter(dataloader))
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), 
                             subplot_kw={'xticks':[], 'yticks':[], 'frame_on':False})
    
    # Add spacing between the images
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    # Iterate over each axis in order to display images and associated labels
    for ax, img, lbl in zip(axes.ravel(), images, labels): 
        img = unnormalize(img, mean, std)
        img = img.numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(class_names[lbl])
    
    plt.show()