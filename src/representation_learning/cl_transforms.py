import torch
import random
import numpy as np
import cv2
from torchvision import transforms


class CustomColorJitter:
    """
    Applies color jitter to the first 4 channels of each image in a batch.
    Retains the mask channel (if present) without modification.

    Attributes:
        color_jitter (transforms.ColorJitter): 
            Transformation to apply brightness, contrast, saturation, and hue changes.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Initializes the color jitter transformation.

        Parameters:
            brightness (float): Brightness factor.
            contrast (float): Contrast factor.
            saturation (float): Saturation factor.
            hue (float): Hue factor.
        """
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, imgs):
        """
        Applies color jitter to each image in the batch.

        Parameters:
            imgs (Tensor): Batch of images with shape [B, C, H, W].

        Returns:
            Tensor: Batch of jittered images.
        """
        batch_jittered = []
        for img in imgs:
            channels = []
            num_channels = img.shape[0]
            # Apply jitter to the first 4 channels only
            for i in range(min(num_channels, 4)):
                single_channel = img[i].unsqueeze(0)
                jittered_channel = self.color_jitter(single_channel)
                channels.append(jittered_channel.squeeze(0))
            # Retain the mask channel if present
            if num_channels == 5:
                mask_channel = img[4].unsqueeze(0)
                channels.append(mask_channel.squeeze(0))
            jittered_img = torch.stack(channels, dim=0)
            batch_jittered.append(jittered_img)
        return torch.stack(batch_jittered, dim=0)

class Cutout(object):
    """
    Applies random square cutouts (holes) to each image in a batch.

    Attributes:
        n_holes (int): Number of holes to cut out from each image.
        length (int): Length of each square hole.
    """

    def __init__(self, n_holes, length):
        """
        Initializes the cutout transformation.

        Parameters:
            n_holes (int): Number of square holes.
            length (int): Length of each hole's side.
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, imgs):
        """
        Applies cutout to each image in the batch.

        Parameters:
            imgs (Tensor): Batch of images with shape [B, C, H, W].

        Returns:
            Tensor: Batch of images with cutout applied.
        """
        batch_size, _, h, w = imgs.size()
        for i in range(batch_size):
            img = imgs[i]
            mask = np.ones((h, w), np.float32)
            for _ in range(self.n_holes):
                y, x = np.random.randint(h), np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                y2 = np.clip(y + self.length // 2, 0, h)
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1: y2, x1: x2] = 0
            mask = torch.from_numpy(mask).to(imgs.device)
            mask = mask.expand_as(img)
            imgs[i] *= mask
        return imgs

    def __repr__(self):
        return f"{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length})"


class GaussianNoise(object):
    """
    Adds Gaussian noise to each image in a batch.

    Attributes:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Adds Gaussian noise to the input tensor.

        Parameters:
            tensor (Tensor): Batch of images with shape [B, C, H, W].

        Returns:
            Tensor: Batch of images with noise added.
        """
        noise = torch.randn_like(tensor).to(tensor.device) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomErodeDilateTransform:
    """
    Applies random erosion or dilation to the mask channel of a batch.

    Attributes:
        kernel_size (int): Size of the circular kernel.
        iterations (int): Number of erosion/dilation iterations.
    """

    def __init__(self, kernel_size=5, iterations=1):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, tensor):
        """
        Applies erosion or dilation randomly to the mask channel.

        Parameters:
            tensor (Tensor): Batch of images with shape [B, 5, H, W].

        Returns:
            Tensor: Batch of processed images.
        """
        batch_size = tensor.shape[0]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        for i in range(batch_size):
            np_array = (tensor[i, 4].cpu().numpy() * 255).astype(np.uint8)
            if random.random() < 0.5:
                processed_np_array = cv2.dilate(np_array, kernel, iterations=self.iterations)
            else:
                processed_np_array = cv2.erode(np_array, kernel, iterations=self.iterations)
            processed_tensor = torch.from_numpy(processed_np_array.astype(np.float32) / 255).to(tensor.device)
            tensor[i, 4] = processed_tensor
        return tensor

    
class ZeroMask:
    """
    Randomly sets the mask channel of an image to zero.

    Attributes:
        p (float): Probability of setting the mask to zero.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        """
        Applies zeroing of the mask channel based on probability.

        Parameters:
            tensor (Tensor): Batch of images with shape [B, 5, H, W].

        Returns:
            Tensor: Batch with zeroed masks.
        """
        for i in range(tensor.shape[0]):
            if random.random() < self.p:
                tensor[i, 4] = torch.zeros_like(tensor[i, 4])
        return tensor


class OnesMask:
    """
    Randomly sets the mask channel of an image to ones.

    Attributes:
        p (float): Probability of setting the mask to ones.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        """
        Applies setting of the mask channel to ones based on probability.

        Parameters:
            tensor (Tensor): Batch of images with shape [B, 5, H, W].

        Returns:
            Tensor: Batch with ones in the mask channel.
        """
        for i in range(tensor.shape[0]):
            if random.random() < self.p:
                tensor[i, 4] = torch.ones_like(tensor[i, 4])
        return tensor

