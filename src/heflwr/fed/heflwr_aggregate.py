from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import Parameters, FitRes
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate_layer
from ..nn import SUPPORT_LAYER
from ..log import logger


def extract(parameters: Parameters, client_net: nn.Module, server_net: nn.Module) -> Parameters:
    """
    Extracts the parameters of a client model (`client_net`) from the global server model (`server_net`)
    based on a given `Parameters` object.

    This function takes the serialized parameters from the global model, represented by the `Parameters` object,
    and transfers them to the client model's architecture.
    The parameters are deserialized and be extracted so that it can suit the client model's architecture.
    This ensures that the client model aligns with the global model's parameters before local training commences,
    which is a common step in system-heterogeneous federated learning setups.

    :param parameters: A `Parameters` object containing serialized parameters of the global model.
    :param client_net: The neural network model on the client-side that needs to be updated with global parameters.
    :param server_net: The global server-side neural network model from which the parameters are sourced.

    :return: An updated `Parameters` object containing the newly extracted parameters for the client model.
    """
    with torch.no_grad():
        tensor_list: List[torch.Tensor] = [torch.Tensor(_) for _ in parameters_to_ndarrays(parameters)]
        global_dict = {name: tensor for name, tensor in zip(server_net.state_dict().keys(), tensor_list)}
        server_net.load_state_dict(global_dict)

        for layer, father_layer in zip(client_net.modules(), server_net.modules()):
            if isinstance(layer, SUPPORT_LAYER):
                layer.reset_parameters_from_father_layer(father_layer)
            # else:
            #     logger.debug(f"Can't extract {layer}, ignore it.")

        array_list: List[np.ndarray] = [_.numpy() for _ in list(client_net.parameters())]
        parameters = ndarrays_to_parameters(array_list)
    return parameters


def merge(results: List[Tuple[ClientProxy, FitRes]], client_nets: List[nn.Module], server_net: nn.Module) -> Parameters:
    """
    Merges parameters from multiple client models into a global server model.

    This function collects the trained parameters from a list of client models
    and aggregates them into the server model's architecture.
    It is a crucial step in system-heterogeneous federated learning setups where each client
    performs local updates to the model, and the server then integrates these updates.
    Each client model's parameters are weighted by the number of examples it used for training,
    ensuring that the aggregation accounts for the different amounts of data each client has.

    :param results: A list of tuples, each containing a `ClientProxy`
    and its corresponding `FitRes` containing training results.
    :param client_nets: A list of neural network models from each client that participated in the training.
    :param server_net: The global server-side neural network model that will be updated with the aggregated parameters.

    :return: A `Parameters` object containing the aggregated parameters after merging updates from client models.
    """
    with torch.no_grad():
        num_examples_list = []
        parameters_list = []

        for _, fit_res in results:
            num_examples_list.append(fit_res.num_examples)
            array_parameters: List[np.ndarray] = parameters_to_ndarrays(fit_res.parameters)
            tensor_parameters: List[torch.Tensor] = [torch.Tensor(_) for _ in array_parameters]
            parameters_list.append(tensor_parameters)

        for client_net, parameters in zip(client_nets, parameters_list):
            client_dict = {name: tensor for name, tensor in zip(client_net.state_dict().keys(), parameters)}
            client_net.load_state_dict(client_dict)

        for layer_name, layer in dict(server_net.named_modules()).items():
            if isinstance(layer, SUPPORT_LAYER):
                client_layers = [dict(client_net.named_modules())[layer_name] for client_net in client_nets]
                aggregate_layer(layer, client_layers, num_examples_list)
            # else:
            #     logger.debug(f"Can't merge {layer}, ignore it.")

        array_list: List[np.ndarray] = [_.numpy() for _ in list(server_net.parameters())]
        parameters = ndarrays_to_parameters(array_list)
    return parameters
