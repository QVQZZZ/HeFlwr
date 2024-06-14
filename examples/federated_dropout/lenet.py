import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heflwr.nn import SSLinear, SSConv2d


def gen_ranges(neurons, keep_rate):
    keep_indices = np.random.rand(*np.arange(neurons).shape) < keep_rate
    # # 避免全部为False
    # if not keep_indices.any():
    #     keep_indices[np.random.randint(0, neurons)] = True
    true_indices = np.where(keep_indices)[0]
    # 初始化变量
    intervals = []
    start = None

    # 遍历True索引并合并连续区间
    prev: int = -1
    for i in true_indices:
        if start is None:
            start = i
        elif i != prev + 1:
            intervals.append((start, prev + 1))
            start = i
        prev = i

    # 添加最后一个区间
    if start is not None:
        intervals.append((start, prev + 1))

    ret = [(f'{start}/{neurons}', f'{end}/{neurons}') for start, end in intervals]
    if not ret:
        ret = [('0', '1')]
    return ret


class LeNet(nn.Module):
    def __init__(self, p=1, dropout_ins=None) -> None:
        super(LeNet, self).__init__()
        if dropout_ins is not None:
            self.dropout_ins = dropout_ins
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
        else:
            self.dropout_ins = []
            self.dropout_ins.append(gen_ranges(8, p))
            self.conv1 = SSConv2d(1, 8, 5,
                                  in_channels_ranges=('0', '1'), out_channels_ranges=self.dropout_ins[-1])
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout_ins.append(gen_ranges(16, p))
            self.conv2 = SSConv2d(8, 16, 5,
                                  in_channels_ranges=self.dropout_ins[-2], out_channels_ranges=self.dropout_ins[-1])
            self.dropout_ins.append(gen_ranges(120, p))
            self.fc1 = SSLinear(16 * 5 * 5, 120,
                                in_features_ranges=self.dropout_ins[-2], out_features_ranges=self.dropout_ins[-1])
            self.dropout_ins.append(gen_ranges(84, p))
            self.fc2 = SSLinear(120, 84,
                                in_features_ranges=self.dropout_ins[-2], out_features_ranges=self.dropout_ins[-1])
            self.fc3 = SSLinear(84, 10,
                                in_features_ranges=self.dropout_ins[-1], out_features_ranges=('0', '1'))
            self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
