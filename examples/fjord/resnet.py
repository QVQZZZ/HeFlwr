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

    def forward(self, x, keep_ratio=1.0):
        residual = x
        out = MaskedLayer(self.conv1, create_neuron_mask(self.conv1, keep_ratio))(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = MaskedLayer(self.conv2, create_neuron_mask(self.conv2, keep_ratio))(out)
        out = self.bn2(out)
        if self.downsample is not None:
            # residual = self.downsample(x)
            # flatten residual nn.Sequential or overwrite nn.Sequential
            residual = MaskedLayer(self.downsample[0], create_neuron_mask(self.downsample[0], keep_ratio))(x)
            residual = self.downsample[1](residual)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10, p='1'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv = SSConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True,
                             in_channels_ranges=('0', '1'), out_channels_ranges=('0', p))
        self.bn = SSBatchNorm2d(64, features_ranges=('0', p))
        self.relu = nn.ReLU(inplace=True)
        # flatten resnet18 nn.Sequential or overwrite nn.Sequential
        # stage1
        self.layer1_1 = SSBasicBlock(64, 64, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer1_2 = SSBasicBlock(64, 64, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        # stage2
        self.layer2_1 = SSBasicBlock(64, 128, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer2_2 = SSBasicBlock(128, 128, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        # stage3
        self.layer3_1 = SSBasicBlock(128, 256, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer3_2 = SSBasicBlock(256, 256, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        # stage4
        self.layer4_1 = SSBasicBlock(256, 512, stride=2, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.layer4_2 = SSBasicBlock(512, 512, in_channels_ranges=('0', p), out_channels_ranges=('0', p))

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = SSLinear(512, num_classes,
                           in_features_ranges=('0', p), out_features_ranges=('0', '1'))

    def forward(self, x, keep_ratio=1.0):
        out = self.relu(self.bn(MaskedLayer(self.conv, create_neuron_mask(self.conv, keep_ratio))(x)))
        # 第一个阶段
        out = self.layer1_1(out, keep_ratio=keep_ratio)
        out = self.layer1_2(out, keep_ratio=keep_ratio)
        # 第二个阶段
        out = self.layer2_1(out, keep_ratio=keep_ratio)
        out = self.layer2_2(out, keep_ratio=keep_ratio)
        # 第三个阶段
        out = self.layer3_1(out, keep_ratio=keep_ratio)
        out = self.layer3_2(out, keep_ratio=keep_ratio)
        # 第四个阶段
        out = self.layer4_1(out, keep_ratio=keep_ratio)
        out = self.layer4_2(out, keep_ratio=keep_ratio)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(p='1'):
    return ResNet(p=p)


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
