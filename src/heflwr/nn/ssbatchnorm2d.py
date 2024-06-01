from typing import Tuple, List, Union, Optional
from typing_extensions import Self
from fractions import Fraction

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Layer_Range


class SSBatchNorm2d(nn.BatchNorm2d):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            features_ranges: Layer_Range = ('0', '1'),
    ) -> None:

        # if features_ranges belong to Interval, then convert into Intervals.
        if isinstance(features_ranges[0], str):
            features_ranges = [features_ranges]

        # Convert string interval into fraction interval.
        features_ranges = [tuple(map(self.parse_fraction_strings, range_str)) for range_str in features_ranges]

        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__(
            num_features=sum(int(num_features * (end - start)) for start, end in features_ranges),
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            **factory_kwargs
        )

        self.features_ranges = features_ranges
        self.reset_parameters()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def extra_repr(self: Self) -> str:
        base_str = super().extra_repr()
        return (f'{base_str}, '
                f'features_ranges={self.features_ranges}')

    def reset_parameters_from_father_layer(self: Self, father_layer: Union[Self, nn.BatchNorm2d]) -> None:
        """
        Resets the parameters of the current layer based on the parameters of a parent (father) layer.

        This method is used to propagate parameters from a parent layer to the current layer, effectively
        initializing or updating the current layer's parameters.

        The method identifies matching blocks of parameters between the layers based on their relative
        positions, and uses the `get_subset_parameters` and `set_subset_parameters` methods to transfer
        the parameters.

        :param father_layer: The parent layer of type Self or nn.BatchNorm2d
        from which the parameters are to be propagated.

        :return: None. The method updates the parameters of the current layer in place.
        """
        father_indices_start = [int(range_[0] * father_layer.num_features) for range_ in self.features_ranges]
        father_indices_end = [int(range_[1] * father_layer.num_features) for range_ in self.features_ranges]
        father_indices = list(zip(father_indices_start, father_indices_end))
        child_indices = self.convert_indices(father_indices)

        father_to_child = {k: v for k, v in zip(father_indices, child_indices)}

        with torch.no_grad():
            for father_index, child_index in father_to_child.items():
                weight, bias = self.get_subset_parameters(father_layer, father_index)
                self.set_subset_parameters(weight, bias, child_index)

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
    def get_subset_parameters(father_layer: nn.BatchNorm2d,
                              index: List[Tuple[int, int]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        print(f'get_subset_parameters: {index}')
        """
        Retrieves the parameters of a subset of the model's weights and biases.

        This method is used to get a specific block of the model's weight vector and the corresponding bias vector.
        It is particularly useful when dealing with layers of models that need to be
        partially extracted based on certain indices.

        :param father_layer: The nn.BatchNorm2d layer of the model from which the parameters are to be extracted.
        :param index: A tuple specifying the start and end indices in the weight vector to be retrieved.


        :return: A tuple containing the extracted weight tensor and bias tensor.
        """
        weight = father_layer.weight[index[0]: index[1]].clone()
        bias = father_layer.bias[index[0]: index[1]].clone()
        return weight, bias

    def set_subset_parameters(self: Self,
                              weight: torch.Tensor,
                              bias: torch.Tensor,
                              index: Tuple[int, int]) -> None:
        print(f'set_subset_parameters: {index}')
        """
        Sets the parameters of a subset of the model's weights and biases.

        This method is used to update a specific block of the model's weight vector and the corresponding bias vector.
        It is particularly useful when dealing with layers of models that need to be
        partially updated based on certain indices.

        :param weight: A torch.Tensor containing the weight values to be set in the specified subset.
        :param bias: A torch.Tensor containing the bias values to be set in the specified subset.
        :param index: A tuple specifying the start and end index for the weight vector to be updated.

        :return: None. The method updates the weight and bias in-place.
        """
        self.weight[index[0]: index[1]] = weight
        self.bias[index[0]: index[1]] = bias

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


if __name__ == '__main__':
    bn1 = nn.BatchNorm2d(4)
    bn2 = SSBatchNorm2d(4, features_ranges=[('0', '2/4'), ('3/4', '1')])

    new_bias_values = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 每个值都不同
    new_weight_values = torch.tensor([1.1, 1.2, 1.3, 1.4])  # 每个值都不同
    with torch.no_grad():
        bn1.bias = nn.Parameter(new_bias_values)
        bn1.weight = nn.Parameter(new_weight_values)

    bn2.reset_parameters_from_father_layer(bn1)
    print(bn1.weight)
    print(bn1.bias)
    print(bn2.weight)
    print(bn2.bias)