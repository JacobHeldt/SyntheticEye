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

    # Add spacing between images
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # Iterate over each axis to display images and labels
    for ax, img, lbl in zip(axes.ravel(), images, labels):
        # Unnormalize image
        for i in range(3):
            img[i] = img[i] * std[i] + mean[i] 
        img = img.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)  # Ensure values are between 0 and 1
        img = img * 255.0  # Scale the pixel values back to [0, 255] for displaying
        ax.imshow(img.astype(np.uint8))  # Convert to uint8 and display
        ax.set_title(class_names[lbl])

    plt.show()


def check_accuracy(data_loader, model, device, threshold=0.5):
    """
    Calculate and print accuracy of the model on a given DataLoader
    """

    print(type(data_loader))
    correct = 0
    samples = 0

    # Set model to evaluation mode.
    model.eval()

    # Ensure no gradients are being computed during evaluation
    with torch.inference_mode():
        for x, y in tqdm(data_loader):

            # Move the data and labels to the appropriate device
            x = x.to(device=device)
            y = y.to(device=device)

            # Make predictions with the model
            scores = model(x)

            # Convert logits to predictions
            preds = (torch.sigmoid(scores) > threshold).squeeze(1).long()  # The model generally perfomed better on real world problems with a threshold of 0.45

            # Update counters based on models predictions
            correct += (preds == y).sum().item()
            samples += preds.size(0)

        print(f'Got {correct} / {samples} correct with an accuracy of {float(correct)/float(samples)*100:.2f}%')

    # Set model back to train mode
    model.train()

    return correct, samples


def predict_single_image(img_path, model, transforms, device='cuda'):
    """
    Predicts the label for a single image using trained model.
    """
    
    # Load image
    img = Image.open(img_path).convert("RGB")

    # Apply combined transforms
    img_tensor = transforms(img).unsqueeze(0).to(device)

    # Set model to evaluation mode and predict image
    model.eval()
    with torch.inference_mode():
        scores = model(img_tensor)
        probability = torch.sigmoid(scores).squeeze().item()

    # Set the model back to training mode
    model.train()

    # Return computed probability
    return probability


def display_folder_images(folder, model, combined_transforms, num_images=50, device='cuda'):
    """
    Display images from a parent folder along with their correct label and predicted probability.
    This helps us to understand the model
    """
    
    # Possible classes of images
    classes = ['real', 'fake']
    images = []

    # Iterate through classes and get image paths
    for label in classes:
        class_folder = os.path.join(folder, label)
        # Collect every image file path from the current class directory
        img_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
        # Append to all_images list with their label
        images.extend([(img, label) for img in img_files[:num_images]])

    # Initialize grid of subplots to display imagestra
    fig, axis = plt.subplots(5, len(images) // 5, figsize=(25, 15))

    # Display each image with label and predicted probability
    for i, (img_path, correct_label) in enumerate(images):
        # Get predicted probability
        predicted_probability = predict_single_image(img_path, model, combined_transforms)
        
        # Load image with PIL
        img = Image.open(img_path)

        # Display image
        row = i // (len(images) // 5)  
        column = i % (len(images) // 5) 
        axis[row, column].imshow(img)
        axis[row, column].set_title(f"Correct: {correct_label}\nPred: {predicted_probability:.3f}")
        axis[row, column].axis("off")
    
    plt.tight_layout()
    plt.show()