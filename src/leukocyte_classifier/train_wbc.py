import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys


# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.leukocyte_classifier.wbc_classifier import CNNModel
from src.leukocyte_classifier.wbc_dataloader import get_data_loaders

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    """
    Trains a neural network model for a specified number of epochs and evaluates it on the validation set.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        criterion (torch.nn.Module): Loss function to minimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        num_epochs (int): Number of epochs to train the model.
        device (str or torch.device): Device on which to perform training ('cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): The trained model with the best weights.
    """
    # Initialize best model weights and accuracy
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            # Set model mode based on phase
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders['train']
            else:
                model.eval()   # Set model to evaluation mode
                dataloader = dataloaders['val']

            # Iterate over the data in the current phase
            for inputs, labels in dataloader:
                # Move data to the specified device (CPU/GPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass and calculate loss
                with torch.set_grad_enabled(phase == 'train'):  # Enable gradients only in training phase
                    outputs = model(inputs)  # Perform forward pass
                    _, preds = torch.max(outputs, 1)  # Get predictions
                    loss = criterion(outputs, labels)  # Compute loss

                    # Backward pass and optimization (only in training phase)
                    if phase == 'train':
                        loss.backward()  # Backpropagation
                        optimizer.step()  # Update model weights

                # Update running loss and correct predictions count
                running_loss += loss.item() * inputs.size(0)  # Multiply by batch size
                running_corrects += torch.sum(preds == labels.data)  # Count correct predictions

            # Calculate average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            # Print epoch statistics
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Update the best model weights if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load the model with the best weights
    model.load_state_dict(best_model_wts)
    return model


# Main function to set up data loaders, model, and start training
def main(args):
    # Hyperparameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    device = torch.device(args.device)

    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_path=args.data_path)

    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize model, criterion, and optimizer
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), args.output_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WBC classifier model')

    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to training data directory')
    parser.add_argument('--output_weights', type=str,
                      default='./wbc_model.pth',
                      help='Path to save model weights')
    parser.add_argument('--num_epochs', type=int, default=25,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                      help='Learning rate for optimizer')
    parser.add_argument('--device', type=str,
                      help='Device to use for training')

    args = parser.parse_args()

    main(args)