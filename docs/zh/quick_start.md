## 快速开始
本文档将介绍 HeFlwr 的一些最简单易用的使用方法，包括如何监控一个深度学习程序运行时的资源情况，如何建立一个结构化剪枝的神经网络，以及如何利用该神经网络进行异构联邦学习。

本指南只介绍 HeFlwr 组件中最核心最有用的部分。如果您想了解 HeFlwr 的详细信息，请查看完整的 API 参考。

在阅读本指南之前，您应该首先查看 [HeFlwr 主页](https://github.com/QVQZZZ/HeFlwr)
以及 [HeFlwr 安装](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/zh/installation.md)。

## 如何监控一个程序的运行？
为了监控程序的运行，您需要导入 `heflwr.monitor.process_monitor` 下的 `FileMonitor`，它以一种简单的形式进行调用并将监控的结果保存在指定的文件中。
```python
import time

from heflwr.monitor.process_monitor import FileMonitor

def main(second: int = 15):
    time.sleep(second)
    
if __name__ == '__main__':
    monitor = FileMonitor(file="./monitor.log")
    monitor.start()
    main()
    monitor.stop()

    print(monitor.summary())
```
该程序会在控制台输出程序运行过程中系统的一些运行情况，包括 CPU 使用率，内存使用率，网络上下行流量等信息。并在运行目录下创建记录了详细监控信息的
`monitor.log` 文件。探索更多的监控器使用方法，例如：功耗监控、远程客户端监控、Prometheus 监控等，请参考
[`heflwr.monitor` API 文档](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/zh/api/monitor.md)。

## 如何创建一个结构化剪枝的神经网络？
不同于掩码的伪剪枝方案, 结构化剪枝可以显著降低神经网络的训练开销或推理开销。利用
`heflwr.nn` 模块，可以快速构建一个结构化剪枝的神经网络：支持每个单独的网络层都能够自定义剪枝的位置和比例，并且保持和 PyTorch 接近一致的 API。

在下面的示例中，我们用 `SSConv2d` 和 `SSLinear` 创建了一个简单的卷积神经网络（术语 "SS" 是 "SubSet" 的缩写），并用初始化参数 `p`
来控制神经网路中的每层都采用相同的保留度（保留度 = 1 - 剪枝率）。为了使模型能够运行在原任务上，我们需要保持网络的输入和输出维度不变，即网络中第一个卷积层的 `in_channels_ranges`
和网络最后中最后一个线性层的 `out_features_ranges` 始终保持为 `('0', '1')`。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from heflwr.nn import SSLinear, SSConv2d

class CifarCNN(nn.Module):
    def __init__(self, p: str) -> None:
        super(CifarCNN, self).__init__()
        self.conv1 = SSConv2d(3, 8, 5, in_channels_ranges=('0', '1'), out_channels_ranges=('0', p))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = SSConv2d(8, 16, 5, in_channels_ranges=('0', p), out_channels_ranges=('0', p))
        self.fc1 = SSLinear(16 * 5 * 5, 120, in_features_ranges=('0', p), out_features_ranges=('0', p))
        self.fc2 = SSLinear(120, 84, in_features_ranges=('0', p), out_features_ranges=('0', p))
        self.fc3 = SSLinear(84, 10, in_features_ranges=('0', p), out_features_ranges=('0', '1'))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

x = torch.randn([1, 3, 32, 32])    

net1 = CifarCNN(p = "1/4"); y1 = net1(x)
net2 = CifarCNN(p = "2/4"); y2 = net2(x)
net3 = CifarCNN(p = "3/4"); y3 = net3(x)
net4 = CifarCNN(p = "4/4"); y4 = net4(x)
```
该程序创建了 4 种保留度的神经网络，其中 `net4` 未经过剪枝。对于每个剪枝网络层，我们还可以选择多个不同的剪枝位置，并利用它们特殊方法从父网络层种获取相应位置的参数，
探索更多的
探索更多的剪枝使用方法，请参考
[`heflwr.nn` API 文档](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/zh/api/nn.md)。
