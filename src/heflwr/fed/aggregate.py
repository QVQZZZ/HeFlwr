import torch
from typing import Union, List
from ..nn import SSLinear, SSConv2d


# Function to aggregate matrices into the final result matrix with weighted average
def aggregate_weight(ret, matrix, row_index, col_index, weight, total_weights):
    with torch.no_grad():
        for (start, end) in row_index:
            row_start, row_end = int(start * ret.shape[0]), int(end * ret.shape[0])
            for (c_start, c_end) in col_index:
                col_start, col_end = int(c_start * ret.shape[1]), int(c_end * ret.shape[1])
                ret_section = ret[row_start:row_end, col_start:col_end]
                matrix_section = matrix[:(row_end-row_start), :(col_end-col_start)]
                ret[row_start:row_end, col_start:col_end] = (ret_section * total_weights[row_start:row_end, col_start:col_end] + matrix_section * weight) / (total_weights[row_start:row_end, col_start:col_end] + weight)
                total_weights[row_start:row_end, col_start:col_end] += weight


def aggregate_bias(global_bias, bias, row_index, weight, total_weights):
    with torch.no_grad():
        for (start, end) in row_index:
            start_int, end_int = int(start * global_bias.size(0)), int(end * global_bias.size(0))
            global_bias_section = global_bias[start_int:end_int]
            bias_section = bias[:(end_int - start_int)]
            global_bias[start_int:end_int] = (global_bias_section * total_weights[start_int:end_int] + bias_section * weight) / (total_weights[start_int:end_int] + weight)
            total_weights[start_int:end_int] += weight


def aggregate_layer(global_layer: Union[SSLinear, SSConv2d], subset_layers: List[Union[SSLinear, SSConv2d]], weights: List[int]) -> None:
    """
    Advanced aggregation function that directly takes SSConv2d or SSLinear layers.

    Parameters:
    - global_layer: The global layer (either SSLinear or SSConv2d) with parameters to be aggregated into.
    - subset_layers: A list of subset layers (either SSLinear or SSConv2d) with parameters to aggregate.
    - weights: A list of client samples corresponding to each subset layer.
    """
    global_weight, global_bias = global_layer.weight, global_layer.bias
    total_weights_weight = torch.zeros_like(global_weight)
    total_weights_bias = torch.zeros_like(global_bias)

    for subset_layer, client_weight in zip(subset_layers, weights):
        weight, bias = subset_layer.weight, subset_layer.bias
        if isinstance(subset_layer, SSLinear):
            row_index, col_index = subset_layer.out_features_ranges, subset_layer.in_features_ranges
        elif isinstance(subset_layer, SSConv2d):
            row_index, col_index = subset_layer.out_channels_ranges, subset_layer.in_channels_ranges
        else:
            raise TypeError(f'Unsupported type of layer {subset_layer}.')
        aggregate_weight(global_weight, weight, row_index, col_index, client_weight, total_weights_weight)
        aggregate_bias(global_bias, bias, row_index, client_weight, total_weights_bias)

