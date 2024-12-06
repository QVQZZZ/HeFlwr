<div align="center">
    <img src='https://github.com/QVQZZZ/HeFlwr/blob/main/pictures/logo.svg' width="250" alt="logo">
</div>
<h1 align="center"> HeFlwr: A Federated Learning Framework for Heterogeneous Devices </h1>
<div align="center">

English | [简体中文](https://github.com/QVQZZZ/HeFlwr/blob/main/README.zh.md)
</div>

HeFlwr is a framework for deploying federated learning in real-world environments.
It provides a simple interface for researching system heterogeneity in federated learning.
HeFlwr can customize models based on differences in client's computing capabilities and storage capacities,
and monitor their resource usage during training. See our full [quick start](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/en/quick_start.md)!  

> ### The documentation for HeFlwr can be found [here](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/en/home.md).
> 
> ### The Wiki for HeFlwr can be found [here](https://github.com/QVQZZZ/HeFlwr/wiki).

## Introduction📜
Federated Learning uses distributed devices to collaboratively train models while ensuring data privacy.
Federated learning has shown potential in multiple scenarios. However,
the large-scale deployment of federated learning still faces the challenge of system heterogeneity,
i.e., significant differences in computing ability,
storage capacity, network bandwidth, and power consumption limits among various devices.
Numerous efforts have attempted to address this problem in simulated environments,
such as <a href="#heterofl">HeteroFL</a> and <a href="#fedrolex">FedRolex</a>.

HeFlwr aims to provide researchers and developers with a convenient tool for studying system heterogeneity in real-world environments.
HeFlwr's design follows some guiding principles:
- Clean interfaces: The design philosophy of HeFlwr is not to introduce additional learning costs. Its interfaces are largely compatible or similar to PyTorch and Flower.
- Easy customization: HeFlwr provides simple modules that enable researchers to easily customize and manage models suitable for different devices, or reproduce works related to system heterogeneity.
- Resource monitoring: HeFlwr is designed for real environments. Developers can easily deploy federated learning among real devices and monitor the resource usage of these devices.
- Scalability: Many modules of HeFlwr can be expanded or overridden according to actual needs.

## Installation🚀
You can install HeFlwr through `pip`:
``` shell
pip install heflwr
```
To take full advantage of all the features of HeFlwr,
please ensure that PyTorch, Flower and Psutil are correctly installed on your system:
``` shell
pip install flwr
pip install torch torchvision
pip install psutil
```

## Baselines🎉
HeFlwr provides some baseline cases in heterogeneous federated learning (we will add more baselines in the future), using unified parameters and experimental settings to offer a comparison for these baselines:


| Baseline-Accuracy                              | Mnist-IID  | Mnist-NonIID | Cifar10-IID | Cifar10-NonIID |
|------------------------------------------------|------------|--------------|-------------|----------------|
| FedAvg<br/>(Theoretical Upper Bound)           | 99.30%     | 98.88%       | 86.14%      | 82.62%         |
| Federated Dropout                              | 26.32%     | 11.35%       | 16.14%      | 14.05%         |
| HeteroFL                                       | 98.44%     | 91.04%       | 80.71%      | 61.66%         |
| MFL                                            | 98.41%     | 92.43%       | 80.70%      | 66.81%         |
| FedRolex                                       | 97.55%     | 91.17%       | 82.18%      | **67.67%**     |
| FjORD                                          | **98.72%** | **96.82%**   | **83.89%**  | 40.20%         |
| Selective Client<br/>(Theoretical Lower Bound) | 98.67%     | 97.38%       | 80.44%      | 65.43%         |

For the specific settings of the above experiments, including neural network architecture, training hyperparameters (e.g., learning rate, optimizer), federated learning hyperparameters (e.g., communication rounds, number of clients), data partitioning hyperparameters (e.g., Dirichlet distribution alpha), and baseline-specific hyperparameters, please refer to the detailed implementation.

The performance of each baseline may vary depending on the scenario and hyperparameter settings. You can quickly modify these parameters and conduct experiments using the command-line scripts we provide:


- For <a href="#fedavg">FedAvg</a>, see: [FedAvg Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/fedavg/README.md)

- For <a href="#federated_dropout">Federated Dropout</a>，see [Federated Dropout Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/federated_dropout/README.md)

- For <a href="#mfl">MFL</a>，see [MFL Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/mfl/README.md)

- For <a href="#heterofl">HeteroFL</a>, see: [HeteroFL Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/heterofl/README.md)

- For <a href="#fedrolex">FedRolex</a>, see: [FedRolex Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/fedrolex/README.md)

- For <a href="#fjord">Fjord</a>, see: [Fjord Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/fjord/README.md)

## Quick Start
### How to Monitor a Process?
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

### How to Create a Structurally Pruned Neural Network?
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


## References📕
<strong><p id="fedavg">[1] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communicationefficient learning of deep networks from decentralized data. In _20th International Conference on Artificial Intelligence and Statistics_. PMLR, Ft. Lauderdale, FL, USA, 1273–1282.</p></strong>

<strong><p id="federated_dropout">[2] Sebastian Caldas, Jakub Konečny, H Brendan McMahan, and Ameet Talwalkar. 2018. _Expanding the reach of federated learning by reducing client resource requirements_. online. arXiv:1812.07210 [cs.LG]</p></strong>

<strong><p id="mfl">[3] R. Yu and P. Li. 2021. Toward Resource-Efficient Federated Learning in Mobile Edge Computing. _IEEE Network_ 35, 1 (2021), 148–155. https://doi.org/10.1109/MNET.011.2000295</p></strong>

<strong><p id="heterofl">[4] Enmao Diao, Jie Ding, and Vahid Tarokh. 2020. HeteroFL: Computation and communication efficient federated learning for heterogeneous clients. In _International Conference on Learning Representations (ICLR)_, Vol. 1. ICLR, online, 1.</p></strong>

<strong><p id="fedrolex">[5] Samiul Alam, Luyang Liu, Ming Yan, and Mi Zhang. 2022. FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction. In _Advances in Neural Information Processing Systems_, Vol. 35. Curran Associates, Inc., New Orleans, United States, 158–171.</p></strong>

<strong><p id="fjord">[6] Samuel Horvath, Stefanos Laskaridis, Mario Almeida, Ilias Leontiadis, Stylianos Venieris, and Nicholas Lane. 2021. Fjord: Fair and accurate federated learning under heterogeneous targets with ordered dropout. In _Advances in Neural Information Processing Systems_, Vol. 34. NeurIPS, online, 1–12.</p></strong>
