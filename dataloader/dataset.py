import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HAM10000(Dataset):
    def __init__(self, dataset_dir, transform=None, lesion_type_dict=None):
        """
        Class to load the HAM10000 dataset.
        
        Parameters:
        -----------
        dataset_dir         : str
                              Directory of the dataset.
        transform           : torchvision.transforms
                              Transformations to be applied to the images.
        lesion_type_dict    : dict
                              Dictionary of lesion types.
        """
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(self.dataset_dir, "HAM10000_images")
        self.csv_file = os.path.join(self.dataset_dir, "HAM10000_metadata.csv")
        self.data = pd.read_csv(self.csv_file)
        self.transform = transform
        self.lesion_type_dict = lesion_type_dict


    def __getitem__(self, idx):
        """
        Function to get an item from the dataset.

        Parameters:
        -----------
        idx         : int
                      Index of the item to get.

        Returns:
        --------
        image       : torch tensor
                      Image tensor.
        label       : int
                      Label of the image.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.images_dir, self.data['image_id'][idx] + ".jpg")
        image = Image.open(image_name).convert('RGB')  
        label = self.lesion_type_dict[self.data['dx'][idx]]
        if self.transform:
            image = self.transform(image)
        return image, label
    

    def __len__(self):
        """
        Function to get the length of the dataset.

        Returns:
        --------
        len         : int
                      Length of the dataset.
        """
        return len(self.data)
    
