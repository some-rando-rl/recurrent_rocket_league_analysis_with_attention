import torch.nn as nn
import torch.nn.functional as F

class NextGoalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(133, 500)
        self.linear1 = nn.Linear(500,500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 500)
        self.linear4 = nn.Linear(500, 500)
        self.linear5 = nn.Linear(500, 500)
        self.output_layer = nn.Linear(500,2)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        return self.output_layer(x)