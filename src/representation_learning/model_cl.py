import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.representation_learning.cl_transforms import CustomColorJitter

class Encoder(nn.Module):
    """
    Encoder network for extracting latent features from input images.
    This encoder consists of multiple convolutional layers, each followed by
    batch normalization, ReLU activation, and pooling layers. The final representation
    is obtained via adaptive average pooling and a fully connected layer.

    Attributes:
        conv1, conv2, conv3, conv4 (nn.Conv2d): Convolutional layers for feature extraction.
        bn1, bn2, bn3, bn4 (nn.BatchNorm2d): Batch normalization layers for stabilizing training.
        pool (nn.MaxPool2d): Max pooling layer to reduce spatial dimensions.
        adap_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling to generate fixed-size output.
        fc (nn.Linear): Fully connected layer to generate the final latent representation.

    Methods:
        forward(x): Forward pass through the encoder to generate feature representations.
    """

    def __init__(self, input_channels, output_features):
        """
        Initializes the encoder model.

        Parameters:
            input_channels (int): Number of input channels (e.g., number of color channels in an image).
            output_features (int): Number of output features (embedding dimension).
        """
        super(Encoder, self).__init__()

        # First convolutional block: Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional block: Conv -> BN -> ReLU -> Pool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional block: Conv -> BN -> ReLU -> Pool
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth convolutional block: Conv -> BN -> ReLU -> Pool
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling layers for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling to reduce spatial dimensions
        self.adap_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to output 1x1 spatial dimensions

        # Fully connected layer to produce final latent representation
        self.fc = nn.Linear(256, output_features)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Latent feature representation of shape (batch_size, output_features).
        """
        # First block: Conv -> BN -> ReLU -> Pool
        x = F.relu(self.bn1(self.conv1(x)))  # 75 x 75 x 5 -> 75 x 75 x 32
        x = self.pool(x)                    # 75 x 75 x 32 -> 37 x 37 x 32

        # Second block: Conv -> BN -> ReLU -> Pool
        x = F.relu(self.bn2(self.conv2(x)))  # 37 x 37 x 32 -> 37 x 37 x 64
        x = self.pool(x)                    # 37 x 37 x 64 -> 18 x 18 x 64

        # Third block: Conv -> BN -> ReLU -> Pool
        x = F.relu(self.bn3(self.conv3(x)))  # 18 x 18 x 64 -> 18 x 18 x 128
        x = self.pool(x)                    # 18 x 18 x 128 -> 9 x 9 x 128

        # Fourth block: Conv -> BN -> ReLU -> Pool
        x = F.relu(self.bn4(self.conv4(x)))  # 9 x 9 x 128 -> 9 x 9 x 256
        x = self.pool(x)                    # 9 x 9 x 256 -> 4 x 4 x 256

        # Adaptive average pooling to 1x1
        x = self.adap_pool(x)               # 4 x 4 x 256 -> 1 x 1 x 256

        # Flatten the spatial dimensions and pass through the fully connected layer
        x = torch.flatten(x, 1)             # Flatten the 1x1x256 to a 256-dimensional vector
        x = self.fc(x)                      # Final feature representation (batch_size, output_features)

        return x

    

# Contrastive Learning Implementation
class CL(nn.Module):
    """
    Contrastive Learning (CL) model using a customizable encoder and projector network.
    This model follows the SimCLR framework, where the input image is augmented to create
    two views, which are then encoded and projected to a latent space for contrastive loss calculation.

    Attributes:
        encoder (nn.Module): The feature extractor network.
        projector (nn.Sequential): The projection head to map features to latent space.
        h_dim (int): Dimension of the hidden representation from the encoder.
        base_size (int): The base image size after transformation.

    Methods:
        forward(x): Computes the latent representations of two augmented views.
        get_latent(x): Returns the latent representation from the encoder without projection.
    """

    def __init__(self, in_channels=5, h_dim=128, projection_dim=32): 
        """
        Initializes the CL model.

        Parameters:
            in_channels (int): Number of input channels for the encoder (e.g., number of image channels).
            h_dim (int): Dimension of the encoder's output features.
            projection_dim (int): Dimension of the projected latent space.
        """
        super(CL, self).__init__()

        # Encoder network to extract features from input images
        self.encoder = Encoder(input_channels=in_channels, output_features=h_dim)
        self.h_dim = h_dim
        self.base_size = 75

        # Projector network to map the encoder output to the latent space
        self.projector = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),  # Linear layer without bias
            nn.ReLU(),                           # Non-linear activation
            nn.Linear(h_dim, projection_dim, bias=False)  # Final projection layer
        ) 

    # Forward method to compute latent representations
    def forward(self, x):
        """
        Performs forward pass through the CL model.

        Parameters:
            x (torch.Tensor): Input batch of images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - z_i: Projected representation of the first augmented view.
                - z_j: Projected representation of the second augmented view.
                - h_i: Encoder output for the first augmented view.
                - h_j: Encoder output for the second augmented view.
        """

        # Generate two augmented versions of the input
        transform = self.simclr_transform()
        x_i = transform(x)
        x_j = transform(x)

        # Encode both augmented views
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # Project the encoded features to latent space
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return z_i, z_j, h_i, h_j
    
    # Method to get latent representation without projection
    def get_latent(self, x):
        """
        Returns the latent representation of the input image using the encoder.

        Parameters:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Latent representation from the encoder.
        """
        return self.encoder(x)

    

    @staticmethod
    def loss(z_i, z_j, temperature):
        """
        Computes the contrastive loss using normalized embeddings from two views (z_i and z_j).
        The loss follows the InfoNCE formulation commonly used in contrastive learning frameworks.

        Parameters:
            z_i (torch.Tensor): Embeddings from the first view of the batch, of shape (N, D),
                                where N is the batch size and D is the embedding dimension.
            z_j (torch.Tensor): Embeddings from the second view of the batch, of shape (N, D).
            temperature (float): Temperature scaling parameter for contrastive loss.

        Returns:
            torch.Tensor: The contrastive loss as a single scalar tensor.
        """
        # Get the batch size
        N = z_i.size(0)

        # Concatenate the embeddings from both views along the batch dimension (2N, D)
        z = torch.cat((z_i, z_j), dim=0)

        # Normalize the concatenated embeddings along the feature dimension
        z_normed = F.normalize(z, dim=1)

        # Compute the cosine similarity matrix (2N, 2N)
        cosine_similarity_matrix = torch.matmul(z_normed, z_normed.T)

        # Create ground-truth labels for positive pairs
        labels = torch.cat([torch.arange(N) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(z.device)

        # Remove self-similarity from the similarity matrix (diagonal elements)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        cosine_similarity_matrix = cosine_similarity_matrix[~mask].view(cosine_similarity_matrix.shape[0], -1)

        # Extract positive and negative similarities
        positives = cosine_similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = cosine_similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        # Concatenate positives and negatives for the logits
        logits = torch.cat([positives, negatives], dim=1)

        # Create target labels indicating positive pairs (0th column is positive)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)

        # Apply temperature scaling
        logits = logits / temperature

        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

        
    def simclr_transform(self):
        """Constructs the SimCLR data transformation pipeline."""
        transformations = []
        color_jitter = CustomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)  #Adjust the color jitter parameters as needed, these are optimized for changes seen in our data
        transformations.append(transforms.RandomApply([color_jitter], p=0.5)) #apply color jitter with 50% probability
        transformations.append(transforms.RandomRotation(degrees=180)) #rotate the image anywhere between -180 and 180 degrees
        transformations.append(transforms.RandomHorizontalFlip(p=0.5)) #flip the image horizontally with 50% probability
        transformations.append(transforms.RandomVerticalFlip(p=0.5)) #flip the image vertically with 50% probability
        affine=transforms.RandomAffine(degrees=0, translate=(0.2,0.2)) #translate the image by up to 20% in both x and y directions to account for the fact that the cells are not always perfectly centered
        transformations.append(transforms.RandomApply([affine], p=0.5)) #apply affine transformation with 50% probability

        #OPTIONAL TRANSFORMATIONS
        #erode_dilate = RandomErodeDilateTransform(kernel_size=5, iterations=1)
        #transformations.append(transforms.RandomApply([erode_dilate], p=0.5))
        #transformations.append(ZeroMask(p=0.5))
        #transformations.append(OnesMask(p=0.5))

        blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0)) #blur the image with a kernel size of 3 and a sigma between 0.1 and 3.0
        transformations.append(transforms.RandomApply([blur],p=0.75)) #apply blur with 75% probability, as we want to make sure that the model is robust to noise
        random_crop = transforms.RandomResizedCrop(size=self.base_size, scale=(0.5, 1.0)) #crop the image to a random size between 50% and 100% of the original size to account for variance in cell size within class
        transformations.append(transforms.RandomApply([random_crop], p=0.5)) #apply random crop with 50% probability

        #OPTIONAL TRANSFORMATIONS
        #if self.config.use_cutout:
            #transformations.append(Cutout(n_holes=1, length=32))
        #if self.config.use_guassian_noise:
        #transformations.append(GaussianNoise(mean=0.0, std=0.1))

        data_transforms = transforms.Compose(transformations) #combine all the transformations into a single transform
        return data_transforms 


if __name__ == '__main__':
    img = torch.randn(1024, 5, 75, 75).to("cuda:1")
    model = CL(in_channels=5, h_dim=128).to("cuda:1")
    z_i, z_j, h_i, h_j = model(img)
    loss = model.loss(z_i, z_j, temperature=0.5)
    print(loss)
