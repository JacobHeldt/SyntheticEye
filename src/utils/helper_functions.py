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
from collections import Counter


def get_image_mean_std(dataset_path, device='cuda', batch_size=512):
    """
    Compute the mean and standard deviation of images in the given dataset path.
    Return a tuple containing mean and standard deviation of dataset.
    """

    # Define image transforms 
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


def categorize_dimension(dimension, categories):
    for i, (lower_bound, upper_bound) in enumerate(categories):
        if lower_bound <= dimension <= upper_bound:
            return i
    return len(categories)

def plot_image_dimensions_bar_graph(img_dir, heading='Image Dimensions'):
    # Define categories for width and height
    width_categories = [(0, 64), (65, 128), (129, 192), (193, 256), (257, 320), (312, 384), (385, 448), (449, 512), (513, 576), (577, 640), (641, 704), (705, 768), (769, 832), (833, 896), (834, 960), (961, 1024), (1025, 1088), (1089, 1152), (1089, 1152), (1153, 1216), (1217, 1280)] 
    height_categories = [(0, 64), (65, 128), (129, 192), (193, 256), (257, 320), (312, 384), (385, 448), (449, 512), (513, 576), (577, 640), (641, 704), (705, 768), (769, 832), (833, 896), (834, 960), (961, 1024), (1025, 1088), (1089, 1152), (1089, 1152), (1153, 1216), (1217, 1280)] 

    # Dictionaries to store resolutions categories
    width_dimensions = Counter()
    height_dimensions = Counter()

    # Get a list of files
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    # Loop through files in directory
    for file_name in tqdm(files, desc="Processing images"):
        file_path = os.path.join(img_dir, file_name)
        with Image.open(file_path) as img:
            width_category = categorize_dimension(img.width, width_categories)
            height_category = categorize_dimension(img.height, height_categories)

            # Update the dimensions count
            width_dimensions[width_category] += 1
            height_dimensions[height_category] += 1

    # Prepare data for the bar graph
    categories = range(max(len(width_categories), len(height_categories)))
    width_values = [width_dimensions[i] for i in categories]
    height_values = [height_dimensions[i] for i in categories]

    # Plotting
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(categories))

    plt.bar(index, height_values, bar_width, label='Height', color='blue')
    plt.bar(index + bar_width, width_values, bar_width, label='Width', color='orange')

    # Create custom labels for the x-axis
    custom_labels = [f'{w[0]}-{w[1]} px' for w in width_categories]

    plt.xlabel('Pixel Ranges')
    plt.ylabel('Number of Images')
    plt.title(heading)
    plt.xticks(index + bar_width / 2, custom_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_total_image_dimensions_bar_graph(img_dir, heading='Image Dimensions'):
    width_categories = [(0, 64), (65, 128), (129, 192), (193, 256), (257, 320), (312, 384), (385, 448), (449, 512), (513, 576), (577, 640), (641, 704), (705, 768), (769, 832), (833, 896), (834, 960), (961, 1024), (1025, 1088), (1089, 1152), (1153, 1216), (1217, 1280)] 
    height_categories = width_categories  
    
    width_dimensions = Counter()
    height_dimensions = Counter()

    for root, dirs, files in os.walk(img_dir):
        for file_name in tqdm(files, desc="Processing images"):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                file_path = os.path.join(root, file_name)
                with Image.open(file_path) as img:
                    width_category = categorize_dimension(img.width, width_categories)
                    height_category = categorize_dimension(img.height, height_categories)

                    width_dimensions[width_category] += 1
                    height_dimensions[height_category] += 1

    categories = range(max(len(width_categories), len(height_categories)))
    width_values = [width_dimensions.get(i, 0) for i in categories]
    height_values = [height_dimensions.get(i, 0) for i in categories]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(categories))

    plt.bar(index, height_values, bar_width, label='Height', color='blue')
    plt.bar(index + bar_width, width_values, bar_width, label='Width', color='orange')

    custom_labels = [f'{w[0]}-{w[1]} px' for w in width_categories]

    plt.xlabel('Pixel Ranges')
    plt.ylabel('Number of Images')
    plt.title(heading)
    plt.xticks(index + bar_width / 2, custom_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def categorize_dimension(dimension, categories):
    for i, (low, high) in enumerate(categories):
        if low <= dimension <= high:
            return i
    return len(categories)


def plot_class_distribution(img_dir, heading='Class Distribution'):
    # Counter for class distribution
    class_distribution = Counter()

    # Get a list of classes
    classes = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    
    # Count images per class
    for class_name in tqdm(classes, desc="Counting images per class"):
        class_path = os.path.join(img_dir, class_name)
        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        class_distribution[class_name] = num_images

    labels = class_distribution.keys()
    values = class_distribution.values()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='blue')
    plt.xlabel('Class Name')
    plt.ylabel('Number of Images')
    plt.title(heading)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def unnormalize(img, mean, std):
    """
    Unnormalize an image tensor
    """
    img = img.clone() 
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def show_img(dataloader, class_names, mean, std, num_images=24):
    """Display a grid of images from a dataloader with their labels"""
    # Get batch of images with labels from dataloade
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10),
                             subplot_kw={'xticks':[], 'yticks':[], 'frame_on':False})

    # Add spacing between images
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # Iterate over each axis to display images and labels
    for ax, img, lbl in zip(axes.ravel(), images, labels):
        # Unnormalize image for displaying
        for i in range(3):
            img[i] = img[i] * std[i] + mean[i] 
        img = img.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1) 
        img = img * 255.0 
        ax.imshow(img.astype(np.uint8)) 
        ax.set_title(class_names[lbl])

    plt.show()


def check_accuracy(data_loader, model, device='cuda', threshold=0.5):
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

            # Move data and labels to the appropriate device
            x = x.to(device=device)
            y = y.to(device=device)

            # Make predictions with the model
            scores = model(x)

            preds = (torch.sigmoid(scores) > threshold).squeeze(1).long() 

            # Update counters based on models predictions
            correct += (preds == y).sum().item()
            samples += preds.size(0)

        print(f'Got {correct} / {samples} correct with an accuracy of {float(correct)/float(samples)*100:.2f}%')

    # Set model back to train mode
    model.train()

    return correct, samples


def check_accuracy_aletheia4(data_loader, model, device='cuda'):
    correct = 0
    samples = 0
    batch_count = 0

    model.eval()
    with torch.inference_mode():
        for x, y in tqdm(data_loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, preds = torch.max(scores.data, 1)

            correct += (preds == y).sum().item()
            samples += preds.size(0)
            batch_count += 1

            if batch_count % 100 == 0:
                current_accuracy = 100 * correct / samples
                print(f'Current accuracy after {batch_count} batches: {current_accuracy:.2f}%')

        print(f'Final accuracy: {float(correct)/float(samples)*100:.2f}%')

    model.train()
    return correct, samples


def predict_single_image(img_path, model, transforms, device='cuda'):
    """
    Predicts the label for a single image using the trained model.
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
        img_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
        images.extend([(img, label) for img in img_files[:num_images]])

    # Initialize grid of subplots to display images
    fig, axis = plt.subplots(5, len(images) // 5, figsize=(25, 15))

    # Display each image with label and predicted probability
    for i, (img_path, correct_label) in enumerate(images):
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