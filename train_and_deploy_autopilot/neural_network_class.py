import torch.nn as nn

# Define Neural Network
class NeuralNetwork(nn.Module):

    def __init__(self, hidden_layer_sizes):
        super().__init__()
        self.flatten = nn.Flatten()
        modules = []
        modules.append(nn.Linear(3*640*480, hidden_layer_sizes[0]))
        modules.append(nn.ReLU())
        if len(hidden_layer_sizes) > 1:
            for i in range(1, len(hidden_layer_sizes)):
                modules.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
                modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_layer_sizes[-1], 2))
        self.linear_relu_stack = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.flatten(x)
        #print(x[0])
        y_predicted = self.linear_relu_stack(x) 
        return y_predicted # AKA logits
    