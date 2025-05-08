#generate masks from hdf5 file
from cellpose import models
import numpy as np
import h5py
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import tqdm

import sys
sys.path.append('/mnt/deepstore/Final_DeepPhenotyping/')
from src.utils.utils import channels_to_bgr 

mp.set_start_method('spawn', force=True)

def load_h5py(file_path):
    """
    Load images from an HDF5 file.

    Parameters:
        file_path (str): Path to the HDF5 file.

    Returns:
        np.ndarray: Array containing the loaded images.
    """
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]  # Read all images from the file
    return images


def load_model(model_path, device):
    """
    Load a pre-trained Cellpose model for segmentation.

    Parameters:
        model_path (str): Path to the pre-trained Cellpose model file.
        device (str or torch.device): Device to load the model on ('cuda' or 'cpu').

    Returns:
        CellposeModel: Loaded Cellpose model.
    """
    cellpose_model = models.CellposeModel(
        gpu=True, 
        pretrained_model=model_path, 
        device=torch.device(device)
    )
    return cellpose_model


def get_center_mask(arr):
    """
    Generate a binary mask where only the pixels with the same value as the center pixel are kept.

    Parameters:
        arr (np.ndarray): Input array from which to generate the mask.

    Returns:
        np.ndarray: Binary mask where the center region is preserved.
    """
    # Step 1: Get the center pixel's coordinates and value
    center_x = arr.shape[0] // 2
    center_y = arr.shape[1] // 2
    center_value = arr[center_x, center_y]

    # Step 2: Create a mask for pixels equal to the center value
    mask = (arr == center_value)

    # Step 3: Set all other pixels to zero
    arr[~mask] = 0

    # Step 4: Set all non-zero pixels to 1
    arr[mask] = 1

    return arr



def generate_mask(args):
    """
    Generate a binary mask for an image using a pre-trained Cellpose model.

    Parameters:
        args (tuple): A tuple containing:
            - image (np.ndarray): Input image.
            - cp_model (CellposeModel): Pre-trained Cellpose model for segmentation.

    Returns:
        np.ndarray: Binary mask of the input image.
    """
    image, cp_model = args

    # Convert channels to BGR format for processing
    rgb = channels_to_bgr(image, [0, 3], [2, 3], [1, 3])

    # Run the Cellpose model to generate a mask
    mask, _, _ = cp_model.eval(
        rgb, diameter=20, channels=[0, 0], batch_size=8
    )

    # Extract the center mask from the generated mask
    mask = get_center_mask(mask)

    return mask


def main():
    h5_file = '/mnt/deepstore/LBxPheno/train_data/wbc_classifier/all_rare_cells.hdf5'
    model_path = '/mnt/deepstore/LBxPheno/pipeline/model_weights/cellpose_model'
    device = 'cuda'
    cp_model = load_model(model_path, device)
    images = load_h5py(h5_file)
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        masks = list(tqdm.tqdm(executor.map(generate_mask, [(im, cp_model) for im in images])))

    masks = np.array(masks)

    #expand dimensions of masks
    masks = np.expand_dims(masks, axis=-1)

    #convert masks to uint16
    masks = masks.astype(np.uint16)
    #save masks to h5py file
    with h5py.File(h5_file, 'a') as f:
        if 'masks' in f:
            del f['masks']
        f.create_dataset('masks', data=masks)




if __name__ == '__main__':
    main()


