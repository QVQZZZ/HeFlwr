### 准备
开始之前，请确保您的系统已安装以下软件：
- Python 3.6 或更高版本
- pip

### 安装 HeFlwr
使用 pip 运行以下命令安装 HeFlwr 包：
``` shell
pip install heflwr
```

### Installing Flower and PyTorch
为了完整使用 HeFlwr 的功能，还需要安装 Flower、PyTorch 以及 Psutil。按照以下步骤安装它们：

1. 使用 `pip` 安装 Flower：
    ``` shell
    pip install flwr
    ```
    如果您还想启用 Flower 的模拟功能，应运行以下命令：
    ``` shell
    pip install flwr[simulation]
    ```

2. 当前关于异构联邦学习的研究主要集中在传统计算机视觉任务上，因此我们建议安装 PyTorch 以及 torchvision：
    ``` shell
    pip install torch torchvision
    ```
   
3. 为了更好地利用 HeFlwr 资源监控器，还需要下载 Psutil：
   ```
   pip install psutil
   ```

### 验证安装
要验证 HeFlwr、Flower、PyTorch 和 Psutil 是否已成功安装，您可以在终端中运行以下命令：
``` shell
python -c "import heflwr; print(heflwr.__version__)"
python -c "import flwr; print(flwr.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import psutil; print(psutil.__version__)"
```
命令应返回 HeFlwr、Flower、PyTorch 和 Psutil 的版本号，表明它们已成功安装在您的系统上。

### 下一步
成功安装 HeFlwr、Flower、PyTorch 以及 Psutil 后，您可能想要：
- 访问 HeFlwr 的项目[主页](https://github.com/QVQZZZ/HeFlwr)，以总览 HeFlwr 的设计背景和指导理念。
