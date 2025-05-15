import os
import numpy as np
import torch

import torch.nn.functional as F
import pandas as pd
import tifffile
import glob
import h5py
from torch.utils.data import DataLoader, Dataset

import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.representation_learning.model_cl import CL
from src.representation_learning.data_loader import CustomImageDataset

def shuffle_index(label, p=0.5):
    """
    Class-informed shuffling of indices to introduce additional positive pairs
    from instances of the same class. This is useful for contrastive learning.

    Parameters:
        label (np.ndarray or torch.Tensor): Array of class labels.
        p (float): Proportion of positive pairs to generate within each class (default: 0.5).

    Returns:
        np.ndarray: Array of shuffled indices.
    """
    index = np.arange(len(label))  # Initialize index array
    for l in np.unique(label):
        size = int(p * torch.sum(label == l))  # Calculate number of positive pairs
        if size >= 2:
            # Randomly select indices within the class
            t1 = np.random.choice(a=index[label == l], size=size, replace=False)
            t2 = t1.copy()
            np.random.shuffle(t2)  # Shuffle the selected indices
            index[t1] = index[t2]  # Swap positions
    return index



def channels_to_bgr(image, blue_index, green_index, red_index):
    """
    Convert image channels to BGR 3-color format for visualization.

    Parameters:
        image (np.ndarray): Input image with shape (H, W, C) or (1, H, W, C).
        blue_index (list): Indices for blue channels.
        green_index (list): Indices for green channels.
        red_index (list): Indices for red channels.

    Returns:
        np.ndarray: Image in BGR format with combined channels.
    """
    if len(image.shape) == 3:
        image = image[np.newaxis, ...]  # Add batch dimension if not present

    # Initialize BGR array
    bgr = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3), dtype='float')

    # Combine specified channels
    if len(blue_index) != 0:
        bgr[..., 0] = np.sum(image[..., blue_index], axis=-1)
    if len(green_index) != 0:
        bgr[..., 1] = np.sum(image[..., green_index], axis=-1)
    if len(red_index) != 0:
        bgr[..., 2] = np.sum(image[..., red_index], axis=-1)

    # Clip values to maximum allowed by dtype
    max_val = np.iinfo(image.dtype).max
    bgr[bgr > max_val] = max_val
    bgr = bgr.astype(image.dtype)

    return bgr


def load_model(model_path, device):
    """
    Load a pre-trained encoder model from a checkpoint file.

    Parameters:
        model_path (str): Path to the model checkpoint file.
        device (str or torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """
    model = CL(in_channels=5, h_dim=128, projection_dim=64)  # Instantiate the model
    state_dict = torch.load(model_path, map_location=device)['model_state_dict']
    model.load_state_dict(state_dict)  # Load model weights
    model.eval()  # Set model to evaluation mode
    return model


def get_embeddings(model, dataloader, device):
    """
    Extract embeddings from the encoder model for a given dataset.

    Parameters:
        model (torch.nn.Module): Trained model to generate embeddings.
        dataloader (DataLoader): DataLoader object to provide batches of images.
        device (str or torch.device): Device to perform computations.

    Returns:
        torch.Tensor: Concatenated embeddings from the entire dataset.
    """
    model.eval()  # Set model to evaluation mode
    embeddings = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)  # Move data to device
            embeddings.append(model.encoder(x).detach().cpu())  # Store embeddings

    # Concatenate all embedding tensors
    embeddings = torch.cat(embeddings)
    return embeddings



def save_5channel_tiffs_single_file(images, masks, out_file):
    """
    Save a batch of images and masks as a single multi-page 5-channel TIFF file.

    Parameters:
        images (np.ndarray): Array of images with shape (n, h, w, 4).
        masks (np.ndarray): Array of masks with shape (n, h, w, 1).
        out_file (str): Path to the output TIFF file.
    """
    data_5ch = np.concatenate([images, masks], axis=-1)  # Combine images and masks
    data_5ch = np.transpose(data_5ch, (0, 3, 1, 2))  # Convert to (n, c, h, w)

    # Save as a multi-page TIFF file
    tifffile.imwrite(
        out_file,
        data_5ch,
        imagej=True,
        metadata={"axes": "TCYX"}
    )
    print(f"Saved multi-page TIFF: {out_file}")




def save_5channel_tiffs(images, masks, outfiles):
    """
    Save each image as a separate 5-channel TIFF file.

    Parameters:
        images (np.ndarray): Array of images (n, h, w, c).
        masks (np.ndarray): Array of masks (n, h, w, 1).
        outfiles (list of str): List of output file paths.
    """
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError("Expected 4D arrays for images and masks.")

    if images.shape[:3] != masks.shape[:3]:
        raise ValueError("Data and mask shapes do not match.")

    data = np.concatenate([images, masks], axis=-1)  # Combine image and mask channels

    for i in range(data.shape[0]):
        img_5ch = np.transpose(data[i], (2, 0, 1))  # Convert to (c, h, w)
        tifffile.imwrite(outfiles[i], img_5ch, imagej=True, metadata={"axes": "CYX"})



def get_data_loaders(data_path, batch_size=64):
    """
    Create a DataLoader from a given directory containing HDF5 files.

    Parameters:
        data_path (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: DataLoader, images, masks, labels.
    """
    types = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    images_list, masks_list, labels_list = [], [], []

    for label, t in enumerate(types):
        current_type_path = os.path.join(data_path, t)
        current_type_files = glob.glob(os.path.join(current_type_path, "*.hdf5"))

        class_images, class_masks = [], []
        for file_path in current_type_files:
            with h5py.File(file_path, 'r') as f:
                imgs = np.array(f['images'][:])
                msks = np.array(f['masks'][:])
                class_images.append(imgs)
                class_masks.append(msks)

        class_images = np.concatenate(class_images, axis=0)
        class_masks = np.concatenate(class_masks, axis=0)

        class_masks[class_masks > 0] = 1
        images_list.append(class_images)
        masks_list.append(class_masks)
        labels_list.append(np.full(len(class_images), label, dtype=np.int64))

    images = np.concatenate(images_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    dataset = CustomImageDataset(images, masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, images, masks, labels


#write main

if __name__ == "__main__":
	#load model
	model_path = "/mnt/deepstore/LBxPheno/pipeline/model_weights/representation_model.pth"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = load_model(model_path, device)
