import torch
import wandb
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import os
import pandas as pd
import yaml

import sys
sys.path.append('/mnt/deepstore/Final_DeepPhenotyping/')
from src.representation_learning.model_cl import CL as Model
from src.representation_learning.data_loader import get_data_loaders



class Trainer(object):
    """
    Trainer class to manage the training, validation, and model saving process.
    Uses WandB for experiment tracking and logging.

    Attributes:
        main_config (dict): Configuration dictionary containing training settings.
        best_pred (float): Best validation loss observed during training.
        best_epoch (int): Epoch at which the best validation loss occurred.
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.

    Methods:
        run_training(): Executes the entire training pipeline.
        build_dataset(config): Loads the training and validation data.
        build_optimizer(config): Sets up the optimizer based on configuration.
        build_scheduler(config): Configures the learning rate scheduler.
        build_model(config): Instantiates the model to be trained.
        train(config): Trains the model for the specified number of epochs.
        validate(temp): Evaluates the model on the validation dataset.
        save_checkpoint(model, epoch, dir): Saves the model checkpoint.
    """

    def __init__(self, config):
        """
        Initializes the Trainer object.

        Parameters:
            config (dict): Configuration dictionary with model and training parameters.
        """
        self.main_config = config

    def run_training(self):
        """
        Runs the complete training process including dataset loading, model initialization,
        optimizer configuration, scheduler setup, and model training.
        """
        with wandb.init():
            # Set random seed for reproducibility
            if self.main_config['random_seed'] is not None:
                np.random.seed(self.main_config['random_seed'])
                torch.manual_seed(self.main_config['random_seed'])
                torch.cuda.manual_seed(self.main_config['random_seed'])

            self.best_pred = np.inf  # Initialize best loss to infinity
            self.best_epoch = 0  # Track the best epoch

            # Build the dataset, model, optimizer, and scheduler
            self.build_dataset(config=wandb.config)
            self.build_model(config=wandb.config)
            self.build_optimizer(config=wandb.config)
            self.build_scheduler(config=wandb.config)

            # Start training
            self.train(config=wandb.config)

    def build_dataset(self, config):
        """
        Builds the training and validation datasets.

        Parameters:
            config (dict): Configuration dictionary from WandB.
        """
        print(config)
        self.train_loader, self.val_loader = get_data_loaders(
            data_path=self.main_config['data_path'],
            batch_size=config.batch_size
        )

    def build_optimizer(self, config):
        """
        Initializes the optimizer based on the configuration.

        Parameters:
            config (dict): Configuration dictionary specifying optimizer type and parameters.
        """
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum
            )
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=config.max_lr,
                weight_decay=config.weight_decay
            )

    def build_scheduler(self, config):
        """
        Sets up the learning rate scheduler based on the specified type.

        Parameters:
            config (dict): Configuration dictionary with scheduler settings.
        """
        # Scheduler with linear warmup and exponential decay
        if config.scheduler == 'LambdaLR':
            lr_multiplier = config.max_lr / config.base_lr
            self.scheduler = LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: (
                    ((lr_multiplier - 1) * epoch / config.l_e + 1) 
                    if epoch < config.l_e 
                    else lr_multiplier * (config.l_b ** (epoch - config.l_e))
                )
            )
        elif config.scheduler == "Cyclic":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer=self.optimizer,
                base_lr=config.base_lr,
                max_lr=config.max_lr,
                step_size_up=72,
                mode='exp_range',
                gamma=0.96,
                scale_mode='cycle',
                cycle_momentum=False
            )

    def build_model(self, config):
        """
        Instantiates the model based on the configuration.

        Parameters:
            config (dict): Configuration dictionary with model parameters.
        """
        self.model = Model(
            in_channels=config.in_channels,
            h_dim=config.h_dim,
            projection_dim=config.projection_dim
        ).to(self.main_config['device'])

    def train(self, config):
        """
        Trains the model for a specified number of epochs.

        Parameters:
            config (dict): Configuration dictionary specifying training parameters.
        """
        for epoch in range(config.epochs):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.to(self.main_config['device'])
                self.optimizer.zero_grad()

                # Forward pass and loss calculation
                z_i, z_j, _, _ = self.model(data)
                loss = self.model.loss(z_i, z_j, config.temperature)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                data = data.detach().cpu()  # Free memory

            # Update scheduler and log training loss
            self.scheduler.step()
            train_loss /= len(self.train_loader)
            print(f"Epoch {epoch} Loss: {train_loss}", end="\t")
            wandb.log({"train_loss": train_loss})

            # Early stop if loss becomes NaN
            if np.isnan(train_loss):
                print("Training loss is NaN, stopping.")
                break

            # Validate the model
            val_loss, _ = self.validate(config.temperature)
            wandb.log({"val_loss": val_loss})

            # Save the best model
            if epoch == 49:
                self.best_pred = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(self.model, epoch, self.main_config['model_path'])

    def validate(self, temp):
        """
        Validates the model on the validation dataset.

        Parameters:
            temp (float): Temperature parameter for the contrastive loss.

        Returns:
            Tuple[float, list]: Validation loss and list of latent representations.
        """
        self.model.eval()
        val_loss = 0
        h_i_list = []

        for batch_idx, (data, label) in enumerate(self.val_loader):
            with torch.no_grad():
                data = data.to(self.main_config['device'])
                z_i, z_j, _, _ = self.model(data)
                loss = self.model.loss(z_i, z_j, temp)
                val_loss += loss.item()

                # Extract latent representations
                h_i = self.model.encoder(data).detach().cpu().numpy()
                label = label.cpu().numpy()
                for i in range(h_i.shape[0]):
                    h_data = {str(j): h_i[i][j] for j in range(self.model.h_dim)}
                    h_data["label"] = str(int(label[i]))
                    h_i_list.append(h_data)
                data = data.detach().cpu()

        val_loss /= len(self.val_loader)
        print(f"Validation Loss: {val_loss}")
        return val_loss, h_i_list

    def save_checkpoint(self, model, epoch, dir):
        """
        Saves the model checkpoint.

        Parameters:
            model (torch.nn.Module): The model to save.
            epoch (int): The current epoch number.
            dir (str): Directory to save the checkpoint file.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        fname = os.path.join(dir, f"model_ep_{epoch}_loss_{self.best_pred:.4f}.pth")
        torch.save(checkpoint, fname)



if __name__ == '__main__':
    with open('/mnt/deepstore/LBxPheno/src/representation_learning/config/config.yml') as f:
        config = yaml.safe_load(f)
    with open('/mnt/deepstore/LBxPheno/src/representation_learning/config/sweep_config.yml') as f:
        sweep_config = yaml.safe_load(f)

    os.makedirs(config['model_path'], exist_ok=True)
    os.makedirs(config['output_csv'], exist_ok=True)
    #os.makedirs(config['output_hdf'], exist_ok=True)

    trainer = Trainer(config=config)
    if config['tune']:
        wandb.login(key=config['wandb_key'])
        sweep_id = wandb.sweep(sweep_config, project=config['project_name'])
        wandb.agent(sweep_id, trainer.run_training, count=config['count'])

    else:
        trainer.run_training()
