import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from heflwr.nn import SSLinear, SSConv2d


def generate_combinations(a, b, n):
    # Create a list of all possible combinations
    combinations = list(itertools.product([a, b], repeat=n))
    return combinations


TOP = ('0', '1/2')
BOTTOM = ('1/2', '1')
ALL = ('0', '1')
NUM_DROPOUT_COUNTS = 4
NET_STRUCTS = generate_combinations(TOP, BOTTOM, NUM_DROPOUT_COUNTS)
NUM_TYPES = 2 ** NUM_DROPOUT_COUNTS


class LeNet(nn.Module):
    def __init__(self, net_type) -> None:
        super(LeNet, self).__init__()
        if net_type == NUM_TYPES:
            self.dropout_ins = tuple(ALL for _ in range(NUM_DROPOUT_COUNTS))
        else:
            self.dropout_ins = NET_STRUCTS[net_type]
        i = 0
        self.conv1 = SSConv2d(1, 8, 5,
                              in_channels_ranges=('0', '1'), out_channels_ranges=self.dropout_ins[i])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = SSConv2d(8, 16, 5,
                              in_channels_ranges=self.dropout_ins[i], out_channels_ranges=self.dropout_ins[i := i + 1])
        self.fc1 = SSLinear(16 * 5 * 5, 120,
                            in_features_ranges=self.dropout_ins[i], out_features_ranges=self.dropout_ins[i := i + 1])
        self.fc2 = SSLinear(120, 84,
                            in_features_ranges=self.dropout_ins[i], out_features_ranges=self.dropout_ins[i := i + 1])
        self.fc3 = SSLinear(84, 10,
                            in_features_ranges=self.dropout_ins[i], out_features_ranges=('0', '1'))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
