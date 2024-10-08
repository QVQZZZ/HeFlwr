from typing import List

import torch

from ..nn import SSLinear, SSConv2d, SSBatchNorm1d, SSBatchNorm2d, SSEmbedding
from ..nn import SUPPORT_LAYER


# Function to aggregate weight into the final result weight with weighted average
def aggregate_weight(global_matrix, matrix, row_index, col_index, weight, total_weights):
    """
    Aggregates the weighted parameters between two matrices, updating the target matrix in place.

    This method performs a weighted aggregation of parameters from a source matrix to a target matrix
    based on given row and column indices, as well as the specified weight.

    The method divides the target and source matrices into sections according to the provided row and column indices,
    then updates each section of the target matrix with a weighted average of itself
    and the corresponding section from the source matrix.
    The `total_weights` matrix tracks the cumulative weights applied to each section,
    ensuring the average is calculated correctly across successive updates.

    :param global_matrix: The target matrix to be updated.
    :param matrix: The source matrix whose parameters will be aggregated into `global_matrix`.
    :param row_index: A list of tuples specifying the row start and end indices for the sections to be aggregated.
    :param col_index: A list of tuples specifying the column start and end indices for the sections to be aggregated.
    :param weight: The weight to be applied to the parameters from `matrix` during aggregation.
    :param total_weights: A matrix of the same shape as `global_matrix`, tracking the total weights applied to each section.

    :return: None. The function updates `global_matrix` and `total_weights` in place.
    """
    with (torch.no_grad()):
        for (start, end) in row_index:
            row_start, row_end = int(start * global_matrix.shape[0]), int(end * global_matrix.shape[0])
            for (c_start, c_end) in col_index:
                col_start, col_end = int(c_start * global_matrix.shape[1]), int(c_end * global_matrix.shape[1])
                global_matrix_section = global_matrix[row_start:row_end, col_start:col_end]
                matrix_section = matrix[:(row_end-row_start), :(col_end-col_start)]
                global_matrix[row_start:row_end, col_start:col_end] = (
                    (global_matrix_section * total_weights[row_start:row_end, col_start:col_end]
                     + matrix_section * weight) /
                    (total_weights[row_start:row_end, col_start:col_end] + weight)
                )
                total_weights[row_start:row_end, col_start:col_end] += weight


# Function to aggregate bias into the final result bias with weighted average
def aggregate_bias(global_bias, bias, row_index, weight, total_weights):
    """
    Aggregates the weighted bias between two bias vectors, updating the target bias vector in place.

    This method performs a weighted aggregation of bias values from a source bias vector to a target bias vector
    based on given row indices, as well as the specified weight.

    The method divides the target and source bias vectors into sections according to the provided row indices,
    then updates each section of the target bias vector with a weighted average of itself
    and the corresponding section from the source bias vector.
    The `total_weights` vector tracks the cumulative weights applied to each section,
    ensuring the average is calculated correctly across successive updates.

    :param global_bias: The target bias vector to be updated.
    :param bias: The source bias vector whose values will be aggregated into `global_bias`.
    :param row_index: A list of tuples specifying the row start and end indices for the sections to be aggregated.
    :param weight: The weight to be applied to the bias values from `bias` during aggregation.
    :param total_weights: A vector of the same shape as `global_bias`, tracking the total weights applied to each section.

    :return: None. The function updates `global_bias` and `total_weights` in place.
    """
    with (torch.no_grad()):
        for (start, end) in row_index:
            start_int, end_int = int(start * global_bias.size(0)), int(end * global_bias.size(0))
            global_bias_section = global_bias[start_int:end_int]
            bias_section = bias[:(end_int - start_int)]
            global_bias[start_int:end_int] = (
                (global_bias_section * total_weights[start_int:end_int] + bias_section * weight) /
                (total_weights[start_int:end_int] + weight)
            )
            total_weights[start_int:end_int] += weight


def aggregate_layer(global_layer: SUPPORT_LAYER,
                    subset_layers: List[SUPPORT_LAYER],
                    weights: List[int]) -> None:
    """
    Performs aggregation of layer parameters from multiple subset layers into a global layer.

    This function aggregates parameters (weights and biases) from a collection of subset layers
    into a global layer of the same type.
    The aggregation process involves a weighted update,
    where each subset layer's influence on the global layer is proportional to its associated weight.

    :param global_layer: The global layer that will be updated with aggregated parameters.
    :param subset_layers: A list of subset layers whose parameters will be aggregated into the global layer.
    :param weights: A list of weights corresponding to each subset layer.
    Usually, it represents the number of samples on the client side.

    :return: None. The function updates `global_layer` in place.
    """
    global_weight, global_bias = global_layer.weight, global_layer.bias if hasattr(global_layer, 'bias') else None
    total_weights_weight = torch.zeros_like(global_weight)
    total_weights_bias = torch.zeros_like(global_bias) if global_bias is not None else None

    for subset_layer, client_weight in zip(subset_layers, weights):
        weight, bias = subset_layer.weight, subset_layer.bias if hasattr(subset_layer, 'bias') else None
        if isinstance(subset_layer, (SSBatchNorm1d, SSBatchNorm2d)):
            row_index = subset_layer.features_ranges
            aggregate_bias(global_weight, weight, row_index, client_weight, total_weights_weight)
            aggregate_bias(global_bias, bias, row_index, client_weight, total_weights_bias)
            continue  # Skip weight matrix aggregation since BatchNorm2d only has vector weights
            
        if isinstance(subset_layer, SSLinear):
            row_index, col_index = subset_layer.out_features_ranges, subset_layer.in_features_ranges
            
        elif isinstance(subset_layer, SSConv2d):
            row_index, col_index = subset_layer.out_channels_ranges, subset_layer.in_channels_ranges
        
        # SSEmbedding: keep all num_embeddings, prune embedding_dims
        elif isinstance(subset_layer, SSEmbedding):
            row_index, col_index = [(0, 1)], subset_layer.embedding_dim_ranges
            
        else:
            raise TypeError(f'Unsupported type of layer {subset_layer}.')
            
        # Aggregate weight matrix
        aggregate_weight(global_weight, weight, row_index, col_index, client_weight, total_weights_weight)
        
        # Aggregate bias if present (for SSLinear and SSConv2d)
        # (SSEmbedding only get weight)
        if global_bias and bias:
            aggregate_bias(global_bias, bias, row_index, client_weight, total_weights_bias)


def _distribute(client_net, server_net):
    with torch.no_grad():
        for layer, father_layer in zip(client_net.modules(), server_net.modules()):
            if isinstance(layer, SUPPORT_LAYER):
                layer.reset_parameters_from_father_layer(father_layer)


def _aggregate(weights, client_nets, server_net):
    with torch.no_grad():
        for layer_name, layer in dict(server_net.named_modules()).items():
            if isinstance(layer, SUPPORT_LAYER):
                client_layers = [dict(client_net.named_modules())[layer_name] for client_net in client_nets]
                aggregate_layer(layer, client_layers, weights)
