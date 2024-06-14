from typing import Optional, Tuple, Dict, List, Union
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import Parameters, Scalar, EvaluateRes, EvaluateIns, FitRes, FitIns, GetPropertiesIns
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from heflwr.fed import extract, merge


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class MFL(FedAvg):
    current_client_nets = {}
    current_clients = []

    def __init__(self, *args, network, num_types, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network
        self.num_types = num_types

    def __repr__(self) -> str:
        """ Compute a string representation of the strategy. """
        rep = f"MFL(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """ Initialize global model parameters. """
        net: nn.Module = self.network(net_type=self.num_types)
        arrays: List[np.ndarray] = get_parameters(net)
        return fl.common.ndarrays_to_parameters(arrays)

    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager
                      ) -> List[Tuple[ClientProxy, FitIns]]:
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        self.current_clients = clients

        fit_configurations = []
        server_net = self.network(net_type=self.num_types)
        for client in clients:
            query = GetPropertiesIns({})
            client_id = client.get_properties(query, timeout=30).properties['cid']
            client_net = self.network((client_id + server_round - 1) % self.num_types)
            self.current_client_nets[client_id] = client_net
            p = extract(parameters, client_net, server_net)
            fit_ins = FitIns(parameters=p, config={'net_type': ((client_id + server_round - 1) % self.num_types)})
            fit_configurations.append((client, fit_ins))
        return fit_configurations

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        client_nets = []
        for client, _ in results:
            query = GetPropertiesIns({})
            client_id = client.get_properties(query, timeout=30).properties['cid']
            client_nets.append(self.current_client_nets[client_id])
        parameter_aggregated = merge(results, client_nets, self.network(net_type=self.num_types))
        return parameter_aggregated, {}

    def configure_evaluate(self,
                           server_round: int,
                           parameters: Parameters,
                           client_manager: ClientManager
                           ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = self.current_clients  # 避免evaluate时self.current_client_nets对应的项为空
        evaluate_configurations = []
        server_net = self.network(net_type=self.num_types)
        for client in clients:
            query = GetPropertiesIns({})
            client_id = client.get_properties(query, timeout=30).properties['cid']
            client_net = self.current_client_nets[client_id]
            p = extract(parameters, client_net, server_net)
            evaluate_ins = EvaluateIns(parameters=p, config={})
            evaluate_configurations.append((client, evaluate_ins))
        return evaluate_configurations

    def aggregate_evaluate(self,
                           server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
                           ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
