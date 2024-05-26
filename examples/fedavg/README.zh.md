# FedAvg
## 配置
为了运行本案例，您需要将本目录下载到您的系统中，您可以通过在 shell 或 git bash 中运行以下命令来实现：
``` shell
git clone --depth=1 https://github.com/QVQZZZ/HeFlwr.git \
&& mv HeFlwr/examples/fedavg . \
&& rm -rf HeFlwr \
&& cd fedavg
```
如果您有多个安装了 HeFlwr 的设备，并希望在多个设备之间运行联邦学习。您还需要在另外的设备中运行上述命令。

## 运行
选择一个您喜欢的设备，并运行 `python server.py` 以让它充当联邦学习服务器。

随后在任意设备上，运行 `python client{N}.py` 以让它成为一个联邦学习客户端。
您需要将 {N} 替换为 {1|2|3|4}，所有的数字代表的客户端代码实际上是统一的。在后续的案例中，我们会让 {1|2|3|4} 代表的客户端运行不同的客户端程序。

## 查看结果
在运行服务器的设备终端上可以获取到训练过程中的 loss 和 acc 数据。

在每个设备的运行目录下会生成 `fedavg_test_log.txt` 文件，该文件记录了训练过程中的设备负载信息。
