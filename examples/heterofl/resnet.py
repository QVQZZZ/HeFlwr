import torch
import torch.nn as nn

from heflwr.nn import SSLinear, SSConv2d, SSBatchNorm2d

class SSResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 in_channels_ranges=('0', '1'),
                 out_channels_ranges=('0', '1'),
                 ):
        super(SSResidualBlock, self).__init__()
        self.conv1 = SSConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                              in_channels_ranges=in_channels_ranges,
                              out_channels_ranges=out_channels_ranges)
        self.bn1 = SSBatchNorm2d(out_channels, features_ranges=out_channels_ranges)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                              in_channels_ranges=out_channels_ranges,
                              out_channels_ranges=out_channels_ranges)
        self.bn2 = SSBatchNorm2d(out_channels, features_ranges=out_channels_ranges)
        # 下采样层
        self.downsample = None
        if stride != 1 or in_channels != out_channels:  # stride不是1，或，in_channels不等于out_channels，就下采样
            self.downsample = nn.Sequential(
                SSConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False,
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


class ResNet34(nn.Module):
    def __init__(self, num_classes=10, p='1'):
        super(ResNet34, self).__init__()
        # 输入层
        self.conv1 = SSConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              in_channels_ranges=('0', '1'), out_channels_ranges=('0', p))
        self.bn1 = SSBatchNorm2d(64, features_ranges=('0', p))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stage1
        self.layer1_1 = SSResidualBlock(64, 64, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer1_2 = SSResidualBlock(64, 64, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer1_3 = SSResidualBlock(64, 64, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        # stage2
        self.layer2_1 = SSResidualBlock(64, 128, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer2_2 = SSResidualBlock(128, 128, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer2_3 = SSResidualBlock(128, 128, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer2_4 = SSResidualBlock(128, 128, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        # stage3
        self.layer3_1 = SSResidualBlock(128, 256, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer3_2 = SSResidualBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer3_3 = SSResidualBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer3_4 = SSResidualBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer3_5 = SSResidualBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer3_6 = SSResidualBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        # stage4
        self.layer4_1 = SSResidualBlock(256, 512, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer4_2 = SSResidualBlock(512, 512, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer4_3 = SSResidualBlock(512, 512, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输入：(batch_size, 512, 7, 7)，输出：(batch_size, 512, 1, 1)
        self.fc = SSLinear(512, num_classes, in_features_ranges=('0', p), out_features_ranges=('0', '1'))

    def forward(self, x):
        # 输入层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 第一个阶段
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer1_3(x)
        # 第二个阶段
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer2_4(x)
        # 第三个阶段
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = self.layer3_6(x)
        # 第四个阶段
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_3(x)
        # 输出层
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


x = torch.randn(1, 3, 224, 224)
net1 = ResNet34(p='1/4')
net2 = ResNet34(p='2/4')
net3 = ResNet34(p='3/4')
net4 = ResNet34(p='1')
net1(x)
net2(x)
net3(x)
net4(x)
print("net1")
print(net1)
print("net2")
print(net2)
print("net3")
print(net3)
print("net4")
print(net4)
