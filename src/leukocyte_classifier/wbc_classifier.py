import torch
import torch.nn as nn
import torch.optim as optim

# For image reading and processing
import numpy as np
import os
import sys


class CNNModel(nn.Module):
    """
    A complex convolutional neural network (CNN) architecture for binary classification.
    The model consists of four convolutional blocks followed by fully connected layers.

    Attributes:
        layer1 (nn.Sequential): First convolutional block with Conv2D, BatchNorm, ReLU, and MaxPool.
        layer2 (nn.Sequential): Second convolutional block with Conv2D, BatchNorm, ReLU, and MaxPool.
        layer3 (nn.Sequential): Third convolutional block with Conv2D, BatchNorm, ReLU, and MaxPool.
        layer4 (nn.Sequential): Fourth convolutional block with Conv2D, BatchNorm, ReLU, and MaxPool.
        fc1 (nn.Linear): Fully connected layer with ReLU activation.
        dropout (nn.Dropout): Dropout layer for regularization.
        fc2 (nn.Linear): Output layer for binary classification.

    Methods:
        forward(x): Performs a forward pass through the network and returns the output.
    """

    def __init__(self):
        """
        Initializes the CNN model by defining the convolutional blocks and fully connected layers.
        """
        super(CNNModel, self).__init__()

        # First convolutional block
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),  # Input: 5 channels, Output: 64 channels
            nn.BatchNorm2d(64),                          # Batch normalization for stability
            nn.ReLU(inplace=True),                       # ReLU activation
            nn.MaxPool2d(2)                              # Downsampling with max pooling (kernel size 2)
        )

        # Second convolutional block
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Input: 64 channels, Output: 128 channels
            nn.BatchNorm2d(128),                         # Batch normalization
            nn.ReLU(inplace=True),                       # ReLU activation
            nn.MaxPool2d(2)                              # Max pooling
        )

        # Third convolutional block
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Input: 128 channels, Output: 256 channels
            nn.BatchNorm2d(256),                          # Batch normalization
            nn.ReLU(inplace=True),                        # ReLU activation
            nn.MaxPool2d(2)                               # Max pooling
        )

        # Fourth convolutional block
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Input: 256 channels, Output: 512 channels
            nn.BatchNorm2d(512),                          # Batch normalization
            nn.ReLU(inplace=True),                        # ReLU activation
            nn.MaxPool2d(2)                               # Max pooling
        )

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Flattened input from convolutional layers
        self.dropout = nn.Dropout(0.5)           # Dropout for regularization (50% dropout rate)
        self.fc2 = nn.Linear(1024, 2)            # Output layer for binary classification

    def forward(self, x):
        """
        Performs the forward pass through the CNN model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2).
        """
        # Pass through the convolutional layers
        out = self.layer1(x)  # First block
        out = self.layer2(out)  # Second block
        out = self.layer3(out)  # Third block
        out = self.layer4(out)  # Fourth block

        # Flatten the output from convolutional blocks
        out = out.reshape(out.size(0), -1)  # Flatten to (batch_size, features)

        # Fully connected layers with dropout
        out = self.dropout(nn.functional.relu(self.fc1(out)))  # FC1 with ReLU and dropout

        # Output layer for binary classification
        out = self.fc2(out)

        return out

