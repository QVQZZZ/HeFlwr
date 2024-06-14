import argparse
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import flwr as fl
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

from heflwr.monitor.process_monitor import FileMonitor

from dataloaders import load_partition_data
from lenet import LeNet
from resnet import ResNet18
from utils import DEVICE, train, test


def set_parameters(net, parameters: List[np.ndarray]):
    # 辅助函数，更新网络参数 (nn.Module, List[np.ndarray] -> None)
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# In[federated learning client]
class FlClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train_loader, test_loader, num_examples):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_examples = num_examples

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.train_loader, epochs=1)
        return self.get_parameters(config={}), self.num_examples["train_set"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.test_loader)
        return float(loss), self.num_examples["test_set"], {"accuracy": float(accuracy)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Heflwr baseline client.")
    parser.add_argument('--server_address', type=str, help='The address of fl server.', default='127.0.0.1:8080')
    parser.add_argument('--client_num', type=int, help='The numbers of clients in fl.')
    parser.add_argument('--cid', type=int, help='The id of the client.(Starting from 1)')
    parser.add_argument('--dataset', type=str, help='Dataset name.')
    parser.add_argument('--partition', type=str, help='Dataset partition mode.', choices=['iid', 'noniid'])
    parser.add_argument('--alpha', type=float, help='Dirichlet alpha.', default=0)
    parser.add_argument('--batch_size', type=int, help='Batch_size', default=32)
    args = parser.parse_args()

    server_address = args.server_address
    client_num = args.client_num
    cid = args.cid
    dataset = args.dataset
    partition = args.partition
    alpha = args.alpha
    batch_size = args.batch_size

    if dataset == "cifar10":
        net = ResNet18(p='1').to(DEVICE)
    elif dataset == "mnist":
        net = LeNet(p='1').to(DEVICE)

    if partition == "iid":
        partitioner = IidPartitioner(client_num)
    elif partition == "noniid":
        partitioner = DirichletPartitioner(num_partitions=client_num, partition_by="label",
                                           alpha=alpha, min_partition_size=100, self_balancing=True)
    train_loader, test_loader, num_examples = load_partition_data(dataset, partitioner, cid, batch_size)
    client = FlClient(cid, net, train_loader, test_loader, num_examples)

    monitor = FileMonitor(file='./fedavg_test_log.txt')
    monitor.start()
    print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
    fl.client.start_numpy_client(server_address=server_address, client=client)
    monitor.stop()
