# HeteroFL
## 配置
为了运行本案例，您需要将本目录下载到您的系统中，您可以通过在 shell 或 git bash 中运行以下命令来实现：
``` shell
git clone --depth=1 https://github.com/QVQZZZ/HeFlwr.git \
&& mv HeFlwr/examples/mfl . \
&& rm -rf HeFlwr \
&& cd mfl
```
选择一个您喜欢的设备，运行 `server.py` 以让它充当联邦学习服务器。

随后在任意设备上，运行 `client.py` 以让它成为一个联邦学习客户端。

例如您可以在同一台主机的多个终端上分别运行以下命令：
```shell
python server.py --dataset cifar10 --num_rounds 3
python client.py --server_address 127.0.0.1:8080 --client_num 4 --cid 1 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32
python client.py --server_address 127.0.0.1:8080 --client_num 4 --cid 2 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32
python client.py --server_address 127.0.0.1:8080 --client_num 4 --cid 3 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32
python client.py --server_address 127.0.0.1:8080 --client_num 4 --cid 4 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32
```
这将会使用 ResNet-18 网络对 CIFAR-10 数据集进行联邦学习，总通信轮次为 3，客户端数量为 4，采用 alpha=0.5 的迪利克雷分布对数据集进行分区。

## 查看结果
在运行服务器的设备终端上可以获取到训练过程中的 loss 和 acc 数据。

在每个设备的运行目录下会生成 `mfl_test_log.txt.txt` 文件，该文件记录了训练过程中的设备负载信息。
