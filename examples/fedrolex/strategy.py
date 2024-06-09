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

# from cifarcnn import CifarCNN as Net
from cifarresnet import ResNet18 as Net


map_client_14 = {0: [('0', '1/4')], 1: [('1/4', '2/4')], 2: [('2/4', '3/4')], 3: [('3/4', '1')]}
map_client_24 = {0: [('0', '2/4')], 1: [('1/4', '3/4')], 2: [('2/4', '4/4')], 3: [('0', '1/4'), ('3/4', '1')]}
map_client_34 = {0: [('0', '3/4')], 1: [('1/4', '4/4')], 2: [('0', '1/4'), ('2/4', '1')], 3: [('0', '2/4'), ('3/4', '1')]}
map_client_44 = {0: [('0', '1')], 1: [('0', '1')], 2: [('0', '1')], 3: [('0', '1')]}


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FedRolex(FedAvg):
    def __repr__(self) -> str:
        """ Compute a string representation of the strategy. """
        rep = f"FedRolex(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """ Initialize global model parameters. """
        net: nn.Module = Net(net_struct=('0', '1'))
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

        struct_14 = map_client_14[server_round % 4]
        struct_24 = map_client_24[server_round % 4]
        struct_34 = map_client_34[server_round % 4]
        struct_44 = map_client_44[server_round % 4]

        fit_configurations = []
        server_net = Net(struct_44)
        for client in clients:
            query = GetPropertiesIns({})
            client_id = client.get_properties(query, timeout=30).properties['cid']
            if client_id == 1:
                client_net = Net(struct_14)
                p = extract(parameters, client_net, server_net)
                fit_ins = FitIns(parameters=p, config={'net_struct': str(struct_14)})
                fit_configurations.append((client, fit_ins))
            elif client_id == 2:
                client_net = Net(struct_24)
                p = extract(parameters, client_net, server_net)
                fit_ins = FitIns(parameters=p, config={'net_struct': str(struct_24)})
                fit_configurations.append((client, fit_ins))
            elif client_id == 3:
                client_net = Net(struct_34)
                p = extract(parameters, client_net, server_net)
                fit_ins = FitIns(parameters=p, config={'net_struct': str(struct_34)})
                fit_configurations.append((client, fit_ins))
            elif client_id == 4:
                client_net = Net(struct_44)
                p = extract(parameters, client_net, server_net)
                fit_ins = FitIns(parameters=p, config={'net_struct': str(struct_44)})
                fit_configurations.append((client, fit_ins))
            else:
                raise RuntimeError('Unknown client_id.')
        return fit_configurations

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        struct_14 = map_client_14[server_round % 4]
        struct_24 = map_client_24[server_round % 4]
        struct_34 = map_client_34[server_round % 4]
        struct_44 = map_client_44[server_round % 4]

        client_nets = []
        for client, _ in results:
            query = GetPropertiesIns({})
            client_id = client.get_properties(query, timeout=30).properties['cid']
            if client_id == 1:
                client_nets.append(Net(struct_14))
            elif client_id == 2:
                client_nets.append(Net(struct_24))
            elif client_id == 3:
                client_nets.append(Net(struct_34))
            elif client_id == 4:
                client_nets.append(Net(struct_44))
            else:
                raise RuntimeError('Unknown client_id.')
        parameter_aggregated = merge(results, client_nets, Net(struct_44))
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
        evaluate_configurations = []
        struct_14 = map_client_14[server_round % 4]
        struct_24 = map_client_24[server_round % 4]
        struct_34 = map_client_34[server_round % 4]
        struct_44 = map_client_44[server_round % 4]


        server_net = Net(('0', '1'))
        for client in clients:
            query = GetPropertiesIns({})
            client_id = client.get_properties(query, timeout=30).properties['cid']
            if client_id == 1:
                client_net = Net(struct_14)
                p = extract(parameters, client_net, server_net)
                fit_ins = FitIns(parameters=p, config={})
                evaluate_configurations.append((client, fit_ins))
            elif client_id == 2:
                client_net = Net(struct_24)
                p = extract(parameters, client_net, server_net)
                fit_ins = EvaluateIns(parameters=p, config={})
                evaluate_configurations.append((client, fit_ins))
            elif client_id == 3:
                client_net = Net(struct_34)
                p = extract(parameters, client_net, server_net)
                fit_ins = EvaluateIns(parameters=p, config={})
                evaluate_configurations.append((client, fit_ins))
            elif client_id == 4:
                client_net = Net(struct_44)
                p = extract(parameters, client_net, server_net)
                fit_ins = EvaluateIns(parameters=p, config={})
                evaluate_configurations.append((client, fit_ins))
            else:
                raise RuntimeError('Unknown client_id.')
        return evaluate_configurations

    def aggregate_evaluate(self,
                           server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
                           ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
