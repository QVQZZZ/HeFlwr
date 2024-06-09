# In[resnet]
import numpy as np
import torch
import torch.nn as nn

from heflwr.nn import SSLinear, SSConv2d, SSBatchNorm2d


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


class SSBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 in_channels_ranges=('0', '1'),
                 out_channels_ranges=('0', '1'),
                 ):
        super(SSBasicBlock, self).__init__()
        self.conv1 = SSConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True,
                              in_channels_ranges=in_channels_ranges,
                              out_channels_ranges=out_channels_ranges)
        self.bn1 = SSBatchNorm2d(out_channels, features_ranges=out_channels_ranges)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                              in_channels_ranges=out_channels_ranges,
                              out_channels_ranges=out_channels_ranges)
        self.bn2 = SSBatchNorm2d(out_channels, features_ranges=out_channels_ranges)
        # 下采样层
        self.downsample = None
        if stride != 1 or in_channels != out_channels or True:  # or True: always scaling down sample
            self.downsample = nn.Sequential(
                SSConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True,
                         in_channels_ranges=in_channels_ranges,
                         out_channels_ranges=out_channels_ranges),
                SSBatchNorm2d(out_channels, features_ranges=out_channels_ranges)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, p=1, dropout_ins=None):
        super(ResNet, self).__init__()
        if dropout_ins is not None:
            self.dropout_ins = dropout_ins
            i = 0
            self.in_planes = 64
            self.conv = SSConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True,
                                 in_channels_ranges=('0', '1'), out_channels_ranges=self.dropout_ins[i])
            self.bn = SSBatchNorm2d(64, features_ranges=self.dropout_ins[i])
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                           in_channels_ranges=self.dropout_ins[i], out_channels_ranges=self.dropout_ins[i := i + 1])
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                           in_channels_ranges=self.dropout_ins[i], out_channels_ranges=self.dropout_ins[i := i + 1])
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                           in_channels_ranges=self.dropout_ins[i], out_channels_ranges=self.dropout_ins[i := i + 1])
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                           in_channels_ranges=self.dropout_ins[i], out_channels_ranges=self.dropout_ins[i := i + 1])
            self.avgpool = nn.AvgPool2d(kernel_size=4)
            self.fc = SSLinear(512, num_classes,
                               in_features_ranges=self.dropout_ins[i], out_features_ranges=('0', '1'))
        else:
            self.dropout_ins = []
            self.in_planes = 64

            self.dropout_ins.append(gen_ranges(64, p))
            self.conv = SSConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True,
                                 in_channels_ranges=('0', '1'), out_channels_ranges=self.dropout_ins[-1])
            self.bn = SSBatchNorm2d(64, features_ranges=self.dropout_ins[-1])
            self.relu = nn.ReLU(inplace=True)

            self.dropout_ins.append(gen_ranges(64, p))
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                           in_channels_ranges=self.dropout_ins[-2], out_channels_ranges=self.dropout_ins[-1])

            self.dropout_ins.append(gen_ranges(128, p))
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                           in_channels_ranges=self.dropout_ins[-2], out_channels_ranges=self.dropout_ins[-1])

            self.dropout_ins.append(gen_ranges(256, p))
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                           in_channels_ranges=self.dropout_ins[-2], out_channels_ranges=self.dropout_ins[-1])

            self.dropout_ins.append(gen_ranges(512, p))
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                           in_channels_ranges=self.dropout_ins[-2], out_channels_ranges=self.dropout_ins[-1])
            self.avgpool = nn.AvgPool2d(kernel_size=4)
            self.fc = SSLinear(512, num_classes,
                               in_features_ranges=self.dropout_ins[-1], out_features_ranges=('0', '1'))

    def _make_layer(self, block, planes, num_blocks, stride, in_channels_ranges, out_channels_ranges):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                in_channels_ranges=in_channels_ranges, out_channels_ranges=out_channels_ranges))
            self.in_planes = planes
            in_channels_ranges = out_channels_ranges
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(p=1, dropout_ins=None):
    return ResNet(SSBasicBlock, [2, 2, 2, 2], p=p, dropout_ins=dropout_ins)


def ResNet34(p=1, dropout_ins=None):
    return ResNet(SSBasicBlock, [3, 4, 6, 3], p=p, dropout_ins=dropout_ins)
