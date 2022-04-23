import torch.nn as nn
import torch.nn.functional as F

class NextGoalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(133, 500)
        self.output_layer = nn.Linear(500,2)

    def forward(self, x):
        x=self.activation(self.input_layer(x))
        return self.output_layer(x)