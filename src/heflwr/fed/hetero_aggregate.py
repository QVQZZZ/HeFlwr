import torch
import torch.nn as nn
from typing import List, Union, Tuple
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np
from flwr.common.typing import Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from ..nn import SSLinear, SSConv2d


def extract(parameters: Parameters, client_net: nn.Module, server_net: nn.Module) -> Parameters:
    with torch.no_grad():
        tensor_list: List[torch.Tensor] = [torch.Tensor(_) for _ in parameters_to_ndarrays(parameters)]
        global_dict = {name: tensor for name, tensor in zip(server_net.state_dict().keys(), tensor_list)}
        server_net.load_state_dict(global_dict)
        for layer, father_layer in zip(client_net.modules(), server_net.modules()):
            if hasattr(layer, 'reset_parameters_from_father_layer'):
                layer.reset_parameters_from_father_layer(father_layer)
        # for layer, father_layer in zip(client_net.modules(), server_net.modules()):
        #     try:
        #         layer.reset_parameters_from_father_layer(father_layer)
        #     except AttributeError:
        #         pass
        #     except Exception as e:
        #         raise RuntimeError(e)
        array_list: List[np.ndarray] = [_.numpy() for _ in list(client_net.parameters())]
        parameters = ndarrays_to_parameters(array_list)
    return parameters


def merge(results:List[Tuple[ClientProxy, FitRes]], client_nets: List[nn.Module], server_net: nn.Module) -> Parameters:
    with torch.no_grad():
        from aggregate import aggregate_layer
        num_examples_list = []
        parameters_list = []
        for _, fit_res in results:
            num_examples_list.append(fit_res.num_examples)
            array_parameters: List[np.ndarray] = parameters_to_ndarrays(fit_res.parameters)
            tensor_parameters: List[torch.Tensor] = [torch.Tensor(_) for _ in array_parameters]
            parameters_list.append(tensor_parameters)
        # 还需要用results.para为client_nets赋值
        for client_net, parameters in zip(client_nets, parameters_list):
            client_dict = {name: tensor for name, tensor in zip(client_net.state_dict().keys(), parameters)}
            client_net.load_state_dict(client_dict)
        for layer_name, layer in dict(server_net.named_modules()).items():
            if isinstance(layer, Union[SSConv2d, SSLinear]):
                client_layers = [dict(client_net.named_modules())[layer_name] for client_net in client_nets]
                aggregate_layer(layer, client_layers, num_examples_list)
            else:
                print(layer)
        # for layer_name, layer in server_net._modules.items():
        #     try:
        #         print(layer_name)
        #         client_layers = [client_net._modules[layer_name] for client_net in client_nets]
        #         aggregate_layer(layer, client_layers, num_examples_list)
        #     except Exception as e:
        #         print(e)

        array_list: List[np.ndarray] = [_.numpy() for _ in list(server_net.parameters())]
        parameters = ndarrays_to_parameters(array_list)
    return parameters
