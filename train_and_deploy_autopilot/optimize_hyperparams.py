"""
Script for optimizing the hyperparameters of a NN model using optuna.
Based on: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
"""

import os
from math import floor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

import optuna
from optuna.trial import TrialState

DEVICE = torch.device("cpu")
OUTPUT = 2 

# Define Neural Network
class NeuralNetwork(nn.Module):

    def __init__(self, trial):
        super().__init__()
        self.flatten = nn.Flatten()

        n_hidden_layers = trial.suggest_int("n_layers", 1, 3)
        modules = []

        in_features = 60*80
        # use looping structure of python to define hyperparameters to be tuned
        for i in range(n_hidden_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 500)
            modules.append(nn.Linear(in_features, out_features))
            modules.append(nn.ReLU())
            #p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            #modules.append(nn.Dropout(p))

            in_features = out_features

        modules.append(nn.Linear(in_features, OUTPUT))
        self.linear_relu_stack = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.flatten(x)
        #print(x[0])
        y_predicted = self.linear_relu_stack(x) 
        return y_predicted # AKA logits



# Class for creating a dataset from our collected data
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) / 255 
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return image.float(), steering, throttle


def get_loaders(train_batch_size, test_batch_size):
    # Create a dataset
    annotations_file = "data2022-10-18-16-00/labels.csv"  # the name of the csv file
    img_dir = "data2022-10-18-16-00/images"  # the name of the folder with all the images in it
    collected_data = CustomImageDataset(annotations_file, img_dir)

    train_data_len = len(collected_data)
    train_data_size = floor(train_data_len*0.9)
    test_data_size = round(train_data_len*0.1)

    # Load the datset (split into train and test)
    train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

    return train_dataloader, test_dataloader
    

# Define Training Function
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    
    for batch, (X, steering, throttle) in enumerate(dataloader):
        #Combine steering and throttle into one tensor (2 columns, X rows)
        y = torch.stack((steering, throttle), -1) 
        #y = y.float()
        
        #data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

        # Compute prediction error
        pred = model(X)  # forward propagation
        loss = loss_fn(pred, y)  # compute loss
        optimizer.zero_grad()  # zero previous gradient
        loss.backward()  # back propagatin
        optimizer.step()  # update parameters
        
        """
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        """

        
# Define a test function to evaluate model performance
def test(dataloader, model, loss_fn):
    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, steering, throttle in dataloader:

            #Combine steering and throttle into one tensor (2 columns, X rows)
            y = torch.stack((steering, throttle), -1) 
            y = y.float()
            
            #data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    #print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss



def objective(trial):

    # Generate the model.
    model = NeuralNetwork(trial).to(DEVICE)
    loss_fn = nn.MSELoss()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    train_batch_size = trial.suggest_int("train_bs", 1, 1000)
    test_batch_size = trial.suggest_int("test_bs", 1, 1000)
    train_dataloader, test_dataloader = get_loaders(train_batch_size, test_batch_size)

    EPOCHS = 10

    # Training of the model
    for epoch in range(EPOCHS):
        #print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        loss = test(test_dataloader, model, loss_fn)
        trial.report(loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss



if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))