import torch
import torch.nn.functional as F
from torch import nn as nn



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


