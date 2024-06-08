# In[resnet]
import torch
import torch.nn as nn

from heflwr.nn import SSLinear, SSConv2d, SSBatchNorm2d


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
    def __init__(self, block, num_blocks, num_classes=10, p='1'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv = SSConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True,
                             in_channels_ranges=('0', '1'), out_channels_ranges=('0', p))
        self.bn = SSBatchNorm2d(64, features_ranges=('0', p))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = SSLinear(512, num_classes,
                           in_features_ranges=('0', p), out_features_ranges=('0', '1'))

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


def ResNet18(p='1'):
    return ResNet(SSBasicBlock, [2, 2, 2, 2], p=p)


def ResNet34(p='1'):
    return ResNet(SSBasicBlock, [3, 4, 6, 3], p=p)


# In[flat]
# class ResNet34(nn.Module):
#     def __init__(self, num_classes=10, p='1'):
#         super(ResNet34, self).__init__()
#         # 输入层
#         self.conv1 = SSConv2d(3, 64, kernel_size=3, stride=1, padding=1,
#                               in_channels_ranges=('0', '1'), out_channels_ranges=('0', p))
#         self.bn1 = SSBatchNorm2d(64, features_ranges=('0', p))
#         self.relu = nn.ReLU(inplace=True)
#         # stage1
#         self.layer1_1 = SSBasicBlock(64, 64, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer1_2 = SSBasicBlock(64, 64, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer1_3 = SSBasicBlock(64, 64, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         # stage2
#         self.layer2_1 = SSBasicBlock(64, 128, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer2_2 = SSBasicBlock(128, 128, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer2_3 = SSBasicBlock(128, 128, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer2_4 = SSBasicBlock(128, 128, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         # stage3
#         self.layer3_1 = SSBasicBlock(128, 256, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer3_2 = SSBasicBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer3_3 = SSBasicBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer3_4 = SSBasicBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer3_5 = SSBasicBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer3_6 = SSBasicBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         # stage4
#         self.layer4_1 = SSBasicBlock(256, 512, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer4_2 = SSBasicBlock(512, 512, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         self.layer4_3 = SSBasicBlock(512, 512, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
#         # 输出层
#         self.avgpool = nn.AvgPool2d(kernel_size=4)
#         self.fc = SSLinear(512, num_classes, in_features_ranges=('0', p), out_features_ranges=('0', '1'))
#
#     def forward(self, x):
#         # Input layer
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         # Layer 1
#         x = self.layer1_1(x)
#         x = self.layer1_2(x)
#         x = self.layer1_3(x)
#         # Layer 2
#         x = self.layer2_1(x)
#         x = self.layer2_2(x)
#         x = self.layer2_3(x)
#         x = self.layer2_4(x)
#         # Layer 3
#         x = self.layer3_1(x)
#         x = self.layer3_2(x)
#         x = self.layer3_3(x)
#         x = self.layer3_4(x)
#         x = self.layer3_5(x)
#         x = self.layer3_6(x)
#         # Layer 4
#         x = self.layer4_1(x)
#         x = self.layer4_2(x)
#         x = self.layer4_3(x)
#         # Output layer
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
