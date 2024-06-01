from typing import Tuple, List, Union, Optional
from typing_extensions import Self
from fractions import Fraction

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Layer_Range


class SSConv2d(nn.Conv2d):
    def __init__(self: Self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, bias: bool = True,
                 in_channels_ranges: Layer_Range = ('0', '1'), out_channels_ranges: Layer_Range = ('0', '1')) -> None:

        # if in_channels_ranges/out_channels_ranges belong to Interval, then convert into Intervals.
        if isinstance(in_channels_ranges[0], str):
            in_channels_ranges = [in_channels_ranges]
        if isinstance(out_channels_ranges[0], str):
            out_channels_ranges = [out_channels_ranges]

        # Convert string interval into fraction interval.
        in_channels_ranges = [tuple(map(self.parse_fraction_strings, range_str)) for range_str in in_channels_ranges]
        out_channels_ranges = [tuple(map(self.parse_fraction_strings, range_str)) for range_str in out_channels_ranges]

        super(SSConv2d, self).__init__(
            in_channels=sum(int(in_channels * (end - start)) for start, end in in_channels_ranges),
            out_channels=sum(int(out_channels * (end - start)) for start, end in out_channels_ranges),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.in_channels_ranges = in_channels_ranges
        self.out_channels_ranges = out_channels_ranges
        self.reset_parameters()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self: Self) -> str:
        base_str = super().extra_repr()
        return (f'{base_str}, '
                f'in_channels_ranges={self.in_channels_ranges}, '
                f'out_channels_ranges={self.out_channels_ranges}')

    def reset_parameters_from_father_layer(self: Self, father_layer: Union[Self, nn.Conv2d]) -> None:
        """
        Resets the parameters of the current layer based on the parameters of a parent (father) layer.

        This method is used to propagate parameters from a parent layer to the current layer, effectively
        initializing or updating the current layer's parameters.

        The method identifies matching blocks of parameters between the layers based on their relative
        positions, and uses the `get_subset_parameters` and `set_subset_parameters` methods to transfer
        the parameters.

        :param father_layer: The parent layer of type Self or nn.Conv2d from which the parameters are to be propagated.

        :return: None. The method updates the parameters of the current layer in place.
        """
        father_out_indices_start = [int(out_range[0] * father_layer.out_channels)
                                    for out_range in self.out_channels_ranges]
        father_out_indices_end = [int(out_range[1] * father_layer.out_channels)
                                  for out_range in self.out_channels_ranges]
        father_out_indices = list(zip(father_out_indices_start, father_out_indices_end))

        father_in_indices_start = [int(in_range[0] * father_layer.in_channels)
                                   for in_range in self.in_channels_ranges]
        father_in_indices_end = [int(in_range[1] * father_layer.in_channels)
                                 for in_range in self.in_channels_ranges]
        father_in_indices = list(zip(father_in_indices_start, father_in_indices_end))

        child_out_indices = self.convert_indices(father_out_indices)
        child_in_indices = self.convert_indices(father_in_indices)

        father_to_child_out = {k: v for k, v in zip(father_out_indices, child_out_indices)}
        father_to_child_in = {k: v for k, v in zip(father_in_indices, child_in_indices)}

        with torch.no_grad():
            for father_out_index, child_out_index in father_to_child_out.items():
                for father_in_index, child_in_index in father_to_child_in.items():
                    weight, bias = self.get_subset_parameters(father_layer, father_out_index, father_in_index)
                    self.set_subset_parameters(weight, bias, child_out_index, child_in_index)

    @staticmethod
    def convert_indices(indices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Converts a list of index tuples into a new list of index tuples with adjusted offsets.

        Each tuple in the input list contains a start and end index. This method calculates new start and
        end indices such that each new tuple's start index begins where the previous one ended, effectively
        creating a continuous range of indices without overlap.

        :param indices: A list of tuples, where each tuple contains a pair of integers representing start
                        and end indices.

        :return: A list of tuples with converted indices that maintain the original ranges but with a
                 continuous offset.
        """
        ret_indices = []
        offset = 0
        for idx in indices:
            start, end = idx[0], idx[1]
            ret_start = offset
            ret_end = offset + (end - start)
            ret_indices.append((ret_start, ret_end))
            offset = ret_end
        return ret_indices

    @staticmethod
    def get_subset_parameters(father_layer: nn.Conv2d, out_index: Tuple[int, int],
                              in_index: Tuple[int, int]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieves the parameters of a subset of the model's weights and biases.

        This method is used to get a specific block of the model's weight matrix and the corresponding bias vector.
        It is particularly useful when dealing with layers of models that need to be
        partially extracted based on certain indices.

        :param father_layer: The nn.Conv2d layer of the model from which the parameters are to be extracted.
        :param out_index: A tuple specifying the start and end index
        for the rows in the weight matrix to be retrieved.
        :param in_index: A tuple specifying the start and end index
        for the columns in the weight matrix to be retrieved.

        :return: A tuple containing the extracted weight tensor and bias tensor.
        The bias tensor would be None if the original layer does not have a bias.
        """
        weight = father_layer.weight[out_index[0]: out_index[1], in_index[0]: in_index[1]].clone()
        bias = None
        if father_layer.bias is not None:
            bias = father_layer.bias[out_index[0]: out_index[1]].clone()
        return weight, bias

    def set_subset_parameters(self: Self, weight: torch.Tensor, bias: torch.Tensor,
                              out_index: Tuple[int, int], in_index: Tuple[int, int]) -> None:
        """
        Sets the parameters of a subset of the model's weights and biases.

        This method is used to update a specific block of the model's weight matrix and the corresponding bias vector.
        It is particularly useful when dealing with layers of models that need to be
        partially updated based on certain indices.

        :param weight: A torch.Tensor containing the weight values to be set in the specified subset.
        :param bias: A torch.Tensor containing the bias values to be set in the specified subset.
        If `None`, the bias will not be updated.
        :param out_index: A tuple specifying the start and end index
        for the rows in the weight matrix to be updated.
        :param in_index: A tuple specifying the start and end index
        for the columns in the weight matrix to be updated.

        :return: None. The method updates the weight and bias in-place.
        """
        self.weight[out_index[0]: out_index[1], in_index[0]: in_index[1]] = weight
        if bias is not None:
            self.bias[out_index[0]: out_index[1]] = bias

    @staticmethod
    def parse_fraction_strings(fraction_str: str) -> Fraction:
        """
        A static method that parses a fraction in string format and returns it as a Fraction object.

        The method expects the input string to represent a fraction in the format 'numerator/denominator'.
        Special cases are '0', which returns Fraction(0, 1), and '1', which returns Fraction(1, 1).

        :param fraction_str: A string that represents a fraction in the format 'numerator/denominator'.

        :return: A Fraction object that corresponds to the fraction represented by the input string.
        """
        if fraction_str == '0':
            return Fraction(0, 1)
        if fraction_str == '1':
            return Fraction(1, 1)
        numerator, denominator = map(int, fraction_str.split('/'))
        return Fraction(numerator, denominator)
