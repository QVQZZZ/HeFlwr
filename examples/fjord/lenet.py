import torch
import torch.nn as nn
import torch.nn.functional as F

from heflwr.nn import SSLinear, SSConv2d


class LeNet(nn.Module):
    def __init__(self, p: str = '1') -> None:
        super(LeNet, self).__init__()
        self.conv1 = SSConv2d(1, 8, 5,
                              in_channels_ranges=('0', '1'), out_channels_ranges=('0', p))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = SSConv2d(8, 16, 5,
                              in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.fc1 = SSLinear(16 * 5 * 5, 120,
                            in_features_ranges=('0', p), out_features_ranges=('0', p))
        self.fc2 = SSLinear(120, 84,
                            in_features_ranges=('0', p), out_features_ranges=('0', p))
        self.fc3 = SSLinear(84, 10,
                            in_features_ranges=('0', p), out_features_ranges=('0', '1'))
        self.flatten = nn.Flatten()

    def forward(self, x, keep_ratio=1.0):
        x = self.pool(F.relu(MaskedLayer(self.conv1, create_neuron_mask(self.conv1, keep_ratio))(x)))
        x = self.pool(F.relu(MaskedLayer(self.conv2, create_neuron_mask(self.conv2, keep_ratio))(x)))
        x = self.flatten(x)
        x = F.relu(MaskedLayer(self.fc1, create_neuron_mask(self.fc1, keep_ratio))(x))
        x = F.relu(MaskedLayer(self.fc2, create_neuron_mask(self.fc2, keep_ratio))(x))
        x = self.fc3(x)
        return x


class MaskedLayer(nn.Module):
    def __init__(self, layer, mask):
        super(MaskedLayer, self).__init__()
        device = layer.weight.device
        self.layer = layer
        self.mask = mask.to(device)

    def forward(self, x):
        if isinstance(self.layer, nn.Conv2d):
            masked_weight = self.layer.weight * self.mask[:, None, None, None]
            if self.layer.bias is not None:
                masked_bias = self.layer.bias * self.mask
            else:
                masked_bias = None
            return nn.functional.conv2d(x, masked_weight, masked_bias, self.layer.stride, self.layer.padding)
        if isinstance(self.layer, nn.Linear):
            masked_weight = self.layer.weight * self.mask[:, None]
            if self.layer.bias is not None:
                masked_bias = self.layer.bias * self.mask
            else:
                masked_bias = None
            return nn.functional.linear(x, masked_weight, masked_bias)
        else:
            raise NotImplementedError("该类型层尚未实现")


def create_neuron_mask(layer, keep_ratio):
    if isinstance(layer, nn.Conv2d):
        num_filters = layer.weight.size(0)
        num_kept = int(num_filters * keep_ratio)
        mask = torch.zeros(num_filters)
        mask[:num_kept] = 1.0
    elif isinstance(layer, nn.Linear):
        num_filters = layer.weight.size(0)
        num_kept = int(num_filters * keep_ratio)
        mask = torch.zeros(num_filters)
        mask[:num_kept] = 1.0
    else:
        raise NotImplementedError("该类型层尚未实现")
    return mask

