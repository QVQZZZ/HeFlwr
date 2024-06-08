import torch
import torch.nn as nn
import torch.nn.functional as F

from heflwr.nn import SSLinear, SSConv2d
from heflwr.nn import Layer_Range


class CifarCNN(nn.Module):
    def __init__(self, net_struct: Layer_Range) -> None:
        super(CifarCNN, self).__init__()
        self.conv1 = SSConv2d(3, 8, 5,
                              in_channels_ranges=('0', '1'), out_channels_ranges=net_struct)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = SSConv2d(8, 16, 5,
                              in_channels_ranges=net_struct, out_channels_ranges=net_struct)
        self.fc1 = SSLinear(16 * 5 * 5, 120,
                            in_features_ranges=net_struct, out_features_ranges=net_struct)
        self.fc2 = SSLinear(120, 84,
                            in_features_ranges=net_struct, out_features_ranges=net_struct)
        self.fc3 = SSLinear(84, 10,
                            in_features_ranges=net_struct, out_features_ranges=('0', '1'))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
