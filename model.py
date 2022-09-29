# Pytorch
import torch
import torch.nn as nn

class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, output_dim, bias=False),

            # nn.Linear(input_dim, input_dim, bias=False),
            # nn.ReLU(),
            # nn.Linear(input_dim, input_dim, bias=False),
            # nn.ReLU(),
            # nn.Linear(input_dim, input_dim, bias=False),
            # nn.ReLU(),
            # nn.Linear(input_dim, input_dim, bias=False),
            # nn.ReLU(),
            # nn.Linear(input_dim, input_dim, bias=False),
            # nn.ReLU(),
            # nn.Linear(input_dim, output_dim, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        return x