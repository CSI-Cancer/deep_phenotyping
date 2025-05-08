#data_loader

from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import os
import glob

import torch
import random
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import numbers
import torch.nn as nn
    
class CustomImageDataset(Dataset):
    """
    Custom dataset class for loading and processing 4-channel, 75x75, 16-bit TIFF images.
    Each sample includes a normalized image, a binary mask, and a label. The dataset
    supports applying transformations during data loading.

    Attributes:
        images (np.ndarray): Numpy array of images with shape (N, 4, 75, 75).
        masks (np.ndarray): Numpy array of binary masks with shape (N, 1, 75, 75).
        labels (np.ndarray): Numpy array of labels corresponding to each image.
        tran (bool): Flag to indicate if transformations should be applied.
        t (torchvision.transforms.Compose): Transformation pipeline to convert numpy arrays to tensors.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the image, mask, and label at the given index.
    """

    def __init__(self, images, masks, labels, tran=False):
        """
        Initializes the custom image dataset.

        Parameters:
            images (np.ndarray): Array of input images.
            masks (np.ndarray): Array of binary masks corresponding to the images.
            labels (np.ndarray): Array of integer labels for each image.
            tran (bool): If True, applies transformations during data loading.
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        self.tran = tran

        # Transformation pipeline: Convert numpy arrays to PyTorch tensors
        self.t = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves the image, mask, and label at the given index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - hard_masked_image: The image multiplied by the mask, concatenated with the mask.
                - label: The label corresponding to the image.
        """
        # Normalize the image to the range [0, 1]
        image = self.images[idx].astype(np.float32) / 65535.0
        
        # Retrieve the corresponding label and mask
        label = self.labels[idx]
        mask = self.masks[idx].astype(np.int16)

        # Apply transformation to convert to tensor
        image = self.t(image)
        mask = self.t(mask)

        # Create a masked version of the image by multiplying with the binary mask
        hard_masked_image = image * mask

        # Concatenate the masked image and the mask itself along the channel dimension
        hard_masked_image = torch.cat((hard_masked_image, mask), dim=0)

        return hard_masked_image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(data_path, batch_size=64):
    """
    Creates training and validation data loaders from HDF5 files containing images and masks.
    The function reads data from subdirectories, splits it into training and validation sets,
    and returns data loaders for both sets.

    Parameters:
        data_path (str): Path to the directory containing subfolders with HDF5 files.
        batch_size (int): Number of samples per batch for the data loaders (default: 64).

    Returns:
        Tuple[DataLoader, DataLoader]: 
            - train_loader: DataLoader object for the training set.
            - val_loader: DataLoader object for the validation set.
    """
    # Get subdirectory names (representing classes/individual slides of data) within the data path
    types = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    # Initialize lists to store images, masks, and labels for training and validation
    train_images_list, val_images_list = [], []
    train_masks_list, val_masks_list = [], []
    train_labels_list, val_labels_list = [], []

    print(types)  # Print available types (classes)

    # Iterate over each class/slide folder to load images and masks
    for label, t in enumerate(types):
        print(t)
        current_type_path = os.path.join(data_path, t)

        # Get the list of HDF5 files for the current class/slide
        current_type_files = glob.glob(os.path.join(current_type_path, "*.hdf5"))

        class_images, class_masks = [], []

        # Load images and masks from each HDF5 file
        for file_path in current_type_files:
            with h5py.File(file_path, 'r') as f:
                # Load images and masks, ensuring the correct data types
                imgs = np.array(f['images'][:], dtype=np.float32)
                msks = np.array(f['masks'][:])

                # Append to the current class lists
                class_images.append(imgs)
                class_masks.append(msks)

        # Concatenate all loaded images and masks for the current class/slide
        class_images = np.concatenate(class_images, axis=0)
        class_masks = np.concatenate(class_masks, axis=0)

        # Determine the number of training samples (80% of the dataset)
        num_train = int(len(class_images) * 0.8)

        # Randomly shuffle and split data into training and validation sets
        indices = np.random.permutation(len(class_images))
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        train_imgs, val_imgs = class_images[train_indices], class_images[val_indices]
        train_masks, val_masks = class_masks[train_indices], class_masks[val_indices]

        # Append the training and validation data to the respective lists
        train_images_list.append(train_imgs)
        train_masks_list.append(train_masks)
        train_labels_list.append(np.full(len(train_imgs), label, dtype=np.int64))

        print(len(train_imgs))  # Print the number of training images loaded

        val_images_list.append(val_imgs)
        val_masks_list.append(val_masks)
        val_labels_list.append(np.full(len(val_imgs), label, dtype=np.int64))

    # Concatenate lists from all classes to create the final training and validation sets
    train_images = np.concatenate(train_images_list, axis=0)
    val_images = np.concatenate(val_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    train_masks = np.concatenate(train_masks_list, axis=0)
    val_masks = np.concatenate(val_masks_list, axis=0)

    print(len(train_images), len(val_images))  # Print the number of total training and validation images

    # Create PyTorch datasets using the custom dataset class
    train_dataset = CustomImageDataset(train_images, train_masks, train_labels, tran=True)
    val_dataset = CustomImageDataset(val_images, val_masks, val_labels, tran=False)

    # Create DataLoaders with shuffling for training and no shuffling for validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader,_ = get_data_loaders(data_path="/mnt/deepstore/Final_DeepPhenotyping/train_data/representation_learning/cl_data")
    for x, y in train_loader:
        print(x.shape, y.shape)
        break