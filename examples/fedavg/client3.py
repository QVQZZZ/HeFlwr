# In[import and global vars]
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import flwr as fl

from heflwr.monitor.process_monitor import FileMonitor

from dataloaders import load_data
from cifarcnn import CifarCNN as Net
from utils import DEVICE, train, test

print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
NUM_CLIENTS = 10
BATCH_SIZE = 32


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
    monitor = FileMonitor(file='./fedavg_test_log.txt')
    monitor.start()
    net = Net(p='1').to(DEVICE)
    train_loader, test_loader, num_examples = load_data()
    client = FlClient(3, net, train_loader, test_loader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    monitor.stop()
