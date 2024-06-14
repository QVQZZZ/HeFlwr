# FedRolex
## Setup
To run this example, you need to download this directory to your system. You can do so by running the following command in the shell or git bash:
``` shell
git clone --depth=1 https://github.com/QVQZZZ/HeFlwr.git \
&& mv HeFlwr/examples/fedrolex . \
&& rm -rf HeFlwr \
&& cd fedrolex
```
If you have multiple devices installed with HeFlwr and wish to run federated learning across multiple devices, you need to run the above command on the other devices as well.

To run this example, you need to modify the `server_address` in the `client{N}.py` file and the client IP in `strategy.py`.

## Running
Choose a device you prefer and run `server.py` to let it act as the federated learning server.

Then, on any device, run `client.py` to let it become a federated learning client.

For example, you can run the following commands in multiple terminals on the same machine:
```shell
python server.py --dataset cifar10 --num_rounds 3
python client.py --server_address 127.0.0.1:8080 --client_num 4 --cid 1 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32 --p 1/4
python client.py --server_address 127.0.0.1:8080 --client_num 4 --cid 2 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32 --p 1/2
python client.py --server_address 127.0.0.1:8080 --client_num 4 --cid 3 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32 --p 3/4
python client.py --server_address 127.0.0.1:8080 --client_num 4 --cid 4 --dataset cifar10 --partition noniid --alpha 0.5 --batch_size 32 --p 1
```
This will perform federated learning on the CIFAR-10 dataset using the ResNet-18 network, with a total of 3 communication rounds, 4 clients, and a Dirichlet distribution with alpha=0.5 for data partitioning.
Each client will train a network with different width.

## Results
You can get the training process loss and acc data on the device terminal running the server.

A `fedrolex_test_log.txt` file will be generated in the running directory of each device, which records the device load information during the training process.