# custom_dataset.py
from PIL import Image
from torch.utils.data import Dataset

class ImageLoaderDataset(Dataset):
    """
    A class to load images using PIL and apply the given transform.
    """
    def __init__(self, img_paths, label_list, transform=None):
        self.img_paths = img_paths
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        
        # Convert to RGB if the image is not in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB') 
        
        if self.transform:
            img = self.transform(img)
        
        label = self.label_list[index]
        return img, label
