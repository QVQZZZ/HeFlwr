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


rolling_struct_client_14 = {0: [('0', '1/4')], 1: [('1/4', '2/4')], 2: [('2/4', '3/4')], 3: [('3/4', '1')]}
rolling_struct_client_24 = {0: [('0', '2/4')], 1: [('1/4', '3/4')], 2: [('2/4', '4/4')], 3: [('0', '1/4'), ('3/4', '1')]}
rolling_struct_client_34 = {0: [('0', '3/4')], 1: [('1/4', '4/4')], 2: [('0', '1/4'), ('2/4', '1')], 3: [('0', '2/4'), ('3/4', '1')]}
rolling_struct_client_44 = {0: [('0', '1')], 1: [('0', '1')], 2: [('0', '1')], 3: [('0', '1')]}

def round2config(server_round):
    struct_14 = rolling_struct_client_14[server_round % 4]
    struct_24 = rolling_struct_client_24[server_round % 4]
    struct_34 = rolling_struct_client_34[server_round % 4]
    struct_44 = rolling_struct_client_44[server_round % 4]
    p2struct = {0.25: struct_14, 0.5: struct_24, 0.75: struct_34, 1: struct_44}
    return p2struct

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FedRolex(FedAvg):
    def __init__(self, *args, network, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network

    def __repr__(self) -> str:
        """ Compute a string representation of the strategy. """
        rep = f"FedRolex(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """ Initialize global model parameters. """
        net: nn.Module = self.network(net_struct=('0', '1'))
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

        p2struct = round2config(server_round)
        fit_configurations = []
        server_net = self.network(('0', '1'))
        for client in clients:
            query = GetPropertiesIns({})
            rsp = client.get_properties(query, timeout=30)
            client_p = rsp.properties['p']
            net_struct = p2struct[eval(client_p)]
            client_net = self.network(net_struct)
            p = extract(parameters, client_net, server_net)
            fit_ins = FitIns(parameters=p, config={'net_struct': str(net_struct)})
            fit_configurations.append((client, fit_ins))
        return fit_configurations

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        p2struct = round2config(server_round)
        client_nets = []
        for client, _ in results:
            query = GetPropertiesIns({})
            rsp = client.get_properties(query, timeout=30)
            client_p = rsp.properties['p']
            net_struct = p2struct[eval(client_p)]
            client_nets.append(self.network(net_struct))
        parameter_aggregated = merge(results, client_nets, self.network(('0', '1')))
        return parameter_aggregated, {}

    def configure_evaluate(self,
                           server_round: int,
                           parameters: Parameters,
                           client_manager: ClientManager
                           ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        p2struct = round2config(server_round)
        evaluate_configurations = []
        server_net = self.network(('0', '1'))
        for client in clients:
            query = GetPropertiesIns({})
            rsp = client.get_properties(query, timeout=30)
            client_p = rsp.properties['p']
            net_struct = p2struct[eval(client_p)]
            client_net = self.network(net_struct)
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
