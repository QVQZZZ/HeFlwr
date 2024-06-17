<div align="center">
    <img src='https://github.com/QVQZZZ/HeFlwr/blob/main/pictures/logo.svg' width="250" alt="logo">
</div>
<h1 align="center"> HeFlwr: 用于异构设备的联邦学习框架 </h1>
<div align="center">

[English](https://github.com/QVQZZZ/HeFlwr/blob/main/README.md) | 简体中文
</div>

HeFlwr 是一个用于在真实环境中部署联邦学习的框架。
它为联邦学习中的系统异构性的研究提供简单的接口。
HeFlwr 能根据客户端在计算能力和存储容量等方面的差异定制模型，并在训练过程中监控其资源使用情况。

HeFlwr 的文档可以在[这里](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/zh/home.md)找到。

HeFlwr 的 Wiki 可以在[这里](https://github.com/QVQZZZ/HeFlwr/wiki)找到。

## 简介📜
联邦学习利用分布式的设备协同训练模型，同时确保数据的隐私性，联邦学习已在多个场景中展现了潜力。
然而，大规模部署联邦学习仍着面临系统异构性的挑战，即各设备在计算能力、存储容量、网络带宽和功耗限制等方面存在显著差异。
已有许多工作尝试在模拟环境下解决该问题，如 <a href="#heterofl">HeteroFL</a> 和 <a href="#fedrolex">FedRolex</a>。

HeFlwr 旨在为研究人员和开发者提供一个便利的工具，用于在真实环境中研究系统异构性。HeFlwr 的设计遵循一些指导原则：
- 接口简洁：HeFlwr 的设计理念是不引入额外的学习成本，其接口在很大程度上兼容或类似于 PyTorch 和 Flower。
- 轻松定制：HeFlwr 提供了简洁的模块，使研究人员能够轻松定制和管理适用于不同设备的模型，或复现与系统异构性相关的工作。
- 资源监控：HeFlwr 专为真实环境设计，开发者可以方便地在真实设备之间部署联邦学习，并监控这些设备的资源使用情况。
- 可扩展性：HeFlwr 的许多模块都可以根据实际的需要进行扩展或覆盖。


## 安装🚀
您可以通过 `pip` 来安装 HeFlwr：
``` shell
pip install heflwr
```
为了充分利用 HeFlwr 的所有功能，请确保 PyTorch, Flower 以及 Psutil 已正确安装在您的系统中：
``` shell
pip install flwr
pip install torch torchvision
pip install psutil
```

## 基线🎉
HeFlwr 提供了异构联邦学习中的一些基线案例（未来我们将增加更多的基线），采用统一的参数和实验设置，为这些基线提供了对比：

| Baseline-Accuracy                    | Mnist-IID   | Mnist-NonIID   | Cifar10-IID   | Cifar10-NonIID    |
|--------------------------------------|-------------|----------------|---------------|-------------------|
| FedAvg<br/>(Theoretical Upper Bound) | 99.30%      | 98.88%         | 86.14%        | 82.62%            |
| Federated Dropout                    | 26.32%      | 11.35%         | 16.14%        | 14.05%            |
| HeteroFL                             | **98.44%**  | 91.04%         | 80.71%        | 61.66%            |
| MFL                                  | 98.41%      | **92.43%**     | 80.70%        | 66.81%            |
| FedRolex                             | 97.55%      | 91.17%         | **82.18%**    | **67.67%**        |


上述实验的具体设置，包括神经网络的架构，训练超参数（如学习率，优化器），联邦学习超参数（如通信轮次，客户端数量），数据分区超参数（如狄利克雷分布的 alpha）以及基线特定的超参数，请参考具体的实现。

各基线的表现可能随着场景以及超参数的变化而有所改变，您可以通过我们提供的命令行脚本快速修改这些参数并进行实验：

- 对于 <a href="#fedavg">FedAvg</a>，请查看：[FedAvg Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/fedavg/README.zh.md)

- 对于 <a href="#federated_dropout">Federated Dropout</a>，请查看[Federated Dropout Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/federated_dropout/README.zh.md)

- 对于 <a href="#mfl">MFL</a>，请查看[MFL Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/mfl/README.zh.md)

- 对于 <a href="#heterofl">HeteroFL</a>，请查看：[HeteroFL Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/heterofl/README.zh.md)

- 对于 <a href="#fedrolex">FedRolex</a>，请查看：[FedRolex Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/fedrolex/README.zh.md)

- 对于 <a href="#fjord">Fjord</a>，请查看：[Fjord Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/fjord/README.zh.md)


## 参考文献📕
<strong><p id="fedavg">[1] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communicationefficient learning of deep networks from decentralized data. In _20th International Conference on Artificial Intelligence and Statistics_. PMLR, Ft. Lauderdale, FL, USA, 1273–1282.</p></strong>

<strong><p id="federated_dropout">[2] Sebastian Caldas, Jakub Konečny, H Brendan McMahan, and Ameet Talwalkar. 2018. _Expanding the reach of federated learning by reducing client resource requirements_. online. arXiv:1812.07210 [cs.LG]</p></strong>

<strong><p id="mfl">[3] R. Yu and P. Li. 2021. Toward Resource-Efficient Federated Learning in Mobile Edge Computing. _IEEE Network_ 35, 1 (2021), 148–155. https://doi.org/10.1109/MNET.011.2000295</p></strong>

<strong><p id="heterofl">[4] Enmao Diao, Jie Ding, and Vahid Tarokh. 2020. HeteroFL: Computation and communication efficient federated learning for heterogeneous clients. In _International Conference on Learning Representations (ICLR)_, Vol. 1. ICLR, online, 1.</p></strong>

<strong><p id="fedrolex">[5] Samiul Alam, Luyang Liu, Ming Yan, and Mi Zhang. 2022. FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction. In _Advances in Neural Information Processing Systems_, Vol. 35. Curran Associates, Inc., New Orleans, United States, 158–171.</p></strong>

<strong><p id="fjord">[6] Samuel Horvath, Stefanos Laskaridis, Mario Almeida, Ilias Leontiadis, Stylianos Venieris, and Nicholas Lane. 2021. Fjord: Fair and accurate federated learning under heterogeneous targets with ordered dropout. In _Advances in Neural Information Processing Systems_, Vol. 34. NeurIPS, online, 1–12.</p></strong>