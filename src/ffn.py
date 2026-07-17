import torch
import torch.nn as nn
import math

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x - [N, seq_len, d_model]
        return self.fc2(self.relu(self.fc1(x))) # Output shape - [N, seq_len, d_model] - applies the feed forward neural network on each position in seq_len independently (in parallel) and doesn't mix information.