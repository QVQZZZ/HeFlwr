import torch
import torch.nn as nn

from heflwr.nn import SSLinear, SSConv2d, SSBatchNorm2d
from heflwr.nn import Layer_Range


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
        if stride != 1 or in_channels != out_channels:
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
    def __init__(self, block, num_blocks, num_classes=10, net_struct: Layer_Range = None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv = SSConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True,
                             in_channels_ranges=('0', '1'), out_channels_ranges=net_struct)
        self.bn = SSBatchNorm2d(64, features_ranges=net_struct)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       in_channels_ranges=net_struct, out_channels_ranges=net_struct)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       in_channels_ranges=net_struct, out_channels_ranges=net_struct)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       in_channels_ranges=net_struct, out_channels_ranges=net_struct)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       in_channels_ranges=net_struct, out_channels_ranges=net_struct)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = SSLinear(512, num_classes,
                           in_features_ranges=net_struct, out_features_ranges=('0', '1'))

    def _make_layer(self, block, planes, num_blocks, stride, in_channels_ranges, out_channels_ranges):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                in_channels_ranges=in_channels_ranges, out_channels_ranges=out_channels_ranges))
            self.in_planes = planes
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


def ResNet18(net_struct):
    return ResNet(SSBasicBlock, [2, 2, 2, 2], net_struct=net_struct)


def ResNet34(net_struct):
    return ResNet(SSBasicBlock, [3, 4, 6, 3], net_struct=net_struct)
