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
and monitor their resource usage during training.

The documentation for HeFlwr can be found [here](https://github.com/QVQZZZ/HeFlwr/blob/main/docs/en/home.md).

The Wiki for HeFlwr can be found [here](https://github.com/QVQZZZ/HeFlwr/wiki).


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

## Quick Start🎉
For <a href="#heterofl">HeteroFL</a>, see: [HeteroFL Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/heterofl/README.md)

For <a href="#fedrolex">FedRolex</a>, see: [FedRolex Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/fedrolex/README.md)

For <a href="#fjord">Fjord</a>, see: [Fjord Implementation](https://github.com/QVQZZZ/HeFlwr/blob/main/examples/fjord/README.md)



## References📕
<strong><p id="heterofl">[1] Enmao Diao, Jie Ding, and Vahid Tarokh. 2020. HeteroFL: Computation and communication efficient federated learning for heterogeneous clients. In _International Conference on Learning Representations (ICLR)_, Vol. 1. ICLR, online, 1.</p></strong>

<strong><p id="fedrolex">[2] Samiul Alam, Luyang Liu, Ming Yan, and Mi Zhang. 2022. FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction. In _Advances in Neural Information Processing Systems_, Vol. 35. Curran Associates, Inc., New Orleans, United States, 158–171.</p></strong>

<strong><p id="fjord">[3] Samuel Horvath, Stefanos Laskaridis, Mario Almeida, Ilias Leontiadis, Stylianos Venieris, and Nicholas Lane. 2021. Fjord: Fair and accurate federated learning under heterogeneous targets with ordered dropout. In _Advances in Neural Information Processing Systems_, Vol. 34. NeurIPS, online, 1–12.</p></strong>