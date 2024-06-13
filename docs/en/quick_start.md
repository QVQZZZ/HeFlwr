## Quick Start
This document will introduce some of the simplest and most user-friendly ways to use HeFlwr, including how to monitor the resource usage of a deep learning program at runtime, how to build a structured pruned neural network, and how to utilize this neural network for heterogeneous federated learning.

This guide covers only the most essential and useful parts of the HeFlwr components. For more detailed information about HeFlwr, please refer to the complete API reference.


Before reading this guide, you should first check out the [HeFlwr homepage](https://github.com/QVQZZZ/HeFlwr) and [HeFlwr Installation](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/zh/installation.md).


## How to Monitor a Process?
To monitor a program's runtime, you need to import `FileMonitor` from `heflwr.monitor.process_monitor`. This can be called in a simple manner and the monitoring results will be saved in a specified file.
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
The program will output some runtime system information to the console, including CPU usage, memory usage, and network traffic (both upstream and downstream). It will create a `monitor.log file` in the running directory, which records detailed monitoring information. To explore more monitoring options, such as power consumption monitoring, remote client monitoring, and Prometheus monitoring, please refer to the [`heflwr.monitor` API documentation](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/zh/api/monitor.md).

## How to Create a Structurally Pruned Neural Network?
Unlike pseudo-pruning schemes that use masks, structural pruning can significantly reduce the training or inference overhead of neural networks. Using the `heflwr.nn` module, you can quickly build a structurally pruned neural network: each individual network layer can be customized for pruning location and ratio, and it maintains an API close to that of PyTorch.

In the example below, we use `SSConv2d` and `SSLinear` to create a simple convolutional neural network (the term "SS" stands for "SubSet"), and use the initialization parameter `p` to control the retention rate (retention rate = 1 - pruning rate) for each layer in the network. To ensure the model can run on the original task, we need to keep the input and output dimensions of the network unchanged, i.e., the `in_channels_ranges` of the first convolutional layer and the `out_features_ranges` of the last linear layer should always be ('0', '1').
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
This program creates neural networks with 4 different retention rates, where `net4` is not pruned. For each pruned network layer, we can also choose multiple different pruning locations and use their special methods to obtain parameters for corresponding locations from the parent network layer. To explore more pruning options, please refer to the [`heflwr.nn` API documentation](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/zh/api/nn.md).
