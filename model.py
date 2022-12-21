import torch
from torch import nn
from torch.utils.data import DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 26, hidden_size = 10, output_size = 3):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_relu = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(), 
            nn.Linear(self.hidden_size, self.output_size)

        )
    def forward(self, x):
        f_a = self.linear_relu(x )
        return f_a
    
    
