# Selective Client
## Setup
To run this example, you need to download this directory to your system. You can do so by running the following command in the shell or git bash:
``` shell
git clone --depth=1 https://github.com/QVQZZZ/HeFlwr.git \
&& mv HeFlwr/examples/selective_client . \
&& rm -rf HeFlwr \
&& cd selective_client
```
If you have multiple devices installed with HeFlwr and wish to run federated learning across multiple devices, you need to run the above command on the other devices as well.

## Running
Choose a device you prefer and run `server.py` to let it act as the federated learning server.

Then, on any device, run `client.py` to let it become a federated learning client.

For example, you can run the following commands in multiple terminals on the same machine:
```shell
python server.py --dataset cifar10 --num_rounds 3
python client.py --server_address 127.0.0.1:8080 --client_num 2 --cid 1 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32
python client.py --server_address 127.0.0.1:8080 --client_num 2 --cid 2 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32
```
This will perform federated learning on the CIFAR-10 dataset using the ResNet-18 network, with a total of 3 communication rounds, 4 clients, and a Dirichlet distribution with alpha=0.5 for data partitioning.

Alternatively, you can use the standalone run scripts we provide to quickly reproduce the experiments on the [HeFlwr homepage](https://github.com/QVQZZZ/HeFlwr):
```shell
./selective_client.sh --num_rounds 50 --client_num 8 --dataset mnist --partition noniid --alpha 0.5 --batch_size 64  # Unix shell
.\selective_client.ps1 -num_rounds 50 -client_num 8 -dataset mnist -partition noniid -alpha 0.5 -batch_size 64  # Windows powershell
```


## Results
You can get the training process loss and acc data on the device terminal running the server.

A `selective_client_test_log.txt` file will be generated in the running directory of each device, which records the device load information during the training process.