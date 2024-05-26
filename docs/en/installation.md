### Prerequisites
Before you begin, ensure that you have the following prerequisites installed on your system:
- Python 3.6 or higher
- pip

### Installing HeFlwr
Use pip to install the HeFlwr package by running the following command:
``` shell
pip install heflwr
```

### Installing Flower and PyTorch
For the full functionality of HeFlwr, it is essential to have Flower, PyTorch and Psutil installed. Follow these steps to install:

1. Install Flower using `pip`:
    ``` shell
    pip install flwr
    ```
    If you also want to enable Flower's simulation features, you should run the following command:
    ``` shell
    pip install flwr[simulation]
    ```

2. Current research on heterogeneous federated learning mainly focuses on traditional computer vision tasks, therefore we recommend installing PyTorch and torchvision:
    ``` shell
    pip install torch torchvision
    ```
3. To better utilize the HeFlwr resource monitor, you also need to download Psutil:
    ```
    pip install psutil
    ```

### Verifying the Installation
To verify that HeFlwr, Flower, PyTorch and Psutil have been successfully installed, you can run the following command in your terminal:
``` shell
python -c "import heflwr; print(heflwr.__version__)"
python -c "import flwr; print(flwr.__version__)"
python -c "import torch; print(torch.__version__)"
```
These commands should return the version numbers of HeFlwr, Flower, PyTorch and Psutil, indicating that they have been successfully installed on your system.

### Next Steps
After successfully installing HeFlwr, Flower, PyTorch and Psutil, you may want to:
- Visit HeFlwr's [homepage](https://github.com/QVQZZZ/HeFlwr) for an overview of the design background and guiding principles.