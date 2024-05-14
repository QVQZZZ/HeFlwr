<div align="center">
    <img src='pictures/logo.svg' width="250" alt="logo">
</div>
<h1 align="center"> HeFlwr: A Federated Learning Framework for Heterogeneous Devices </h1>
<div align="center">


English | [简体中文](./README.zh.md)
</div>

HeFlwr is a framework for deploying federated learning in real-world environments.
It provides a simple interface for researching system heterogeneity in federated learning.
HeFlwr can customize models based on differences in client's computing capabilities and storage capacities,
and monitor their resource usage during training.

## 📜Introduction
Federated Learning uses distributed devices to collaboratively train models while ensuring data privacy.
Federated learning has shown potential in multiple scenarios. However,
the large-scale deployment of federated learning still faces the challenge of system heterogeneity,
i.e., significant differences in computing ability,
storage capacity, network bandwidth, and power consumption limits among various devices.
Numerous efforts have attempted to address this problem in simulated environments,
such as <a href="#heterofl">HeteroFL</a> and <a href="#fjord">FjORD</a>.

HeFlwr aims to provide researchers and developers with a convenient tool for studying system heterogeneity in real-world environments.
HeFlwr's design follows some guiding principles:
- Clean interfaces: The design philosophy of HeFlwr is not to introduce additional learning costs. Its interfaces are largely compatible or similar to PyTorch and Flower.
- Easy customization: HeFlwr provides simple modules that enable researchers to easily customize and manage models suitable for different devices, or reproduce works related to system heterogeneity.
- Resource monitoring: HeFlwr is designed for real environments. Developers can easily deploy federated learning among real devices and monitor the resource usage of these devices.
- Scalability: Many modules of HeFlwr can be expanded or overridden according to actual needs.

## 🚀Installation
You can install HeFlwr through `pip`:
``` shell
pip install heflwr
```
To take full advantage of all the features of HeFlwr,
please ensure that PyTorch and Flower are correctly installed on your system:
``` shell
pip install flwr
pip install torch torchvision
```

## 🎉Quick Start
``` python
import heflwr
```

## References
<strong><p id="heterofl">[1] Enmao Diao, Jie Ding, and Vahid Tarokh. Heterofl: Computation and communication efficient federated learning for heterogeneous clients. In _International Conference on Learning Representations_, 2021.</p></strong>
<strong><p id="fjord">[2] Horvath, S., Laskaridis, S., Almeida, M., Leontiadis, I., Venieris, S. I., and Lane, N. D. Fjord: Fair and accurate federated learning under heterogeneous targets with ordered dropout. _35th Conference on Neural Information Processing Systems (NeurIPS)._, 2021.</p></strong>