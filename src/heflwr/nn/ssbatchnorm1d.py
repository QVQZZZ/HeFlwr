from typing import Tuple, List, Union, Optional
from typing_extensions import Self
from fractions import Fraction

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Layer_Range


class SSBatchNorm1d(nn.BatchNorm1d):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = False,
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

        # 计算剪枝后的特征数量
        pruned_num_features = sum(int(num_features * (end - start)) for start, end in features_ranges)

        super().__init__(
            num_features=pruned_num_features,
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

        # momentum 在 ONNX 中仅用于更新
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
            self.running_mean if not bn_training or self.track_running_stats else None,
            self.running_var if not bn_training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def extra_repr(self: Self) -> str:
        base_str = super().extra_repr()
        return (f'{base_str},'
                f'features_ranges={self.features_ranges}')

    def reset_parameters_from_father_layer(self: Self, father_layer: Union[Self, nn.BatchNorm1d]) -> None:
        """
        从父层中复制参数到剪枝后的 BatchNorm1d 层。
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
        将索引区间调整为连续的索引范围。
        """
        ret_indices = []
        offset = 0
        for start, end in indices:
            ret_start = offset
            ret_end = offset + (end - start)
            ret_indices.append((ret_start, ret_end))
            offset = ret_end
        return ret_indices

    @staticmethod
    def get_subset_parameters(father_layer: nn.BatchNorm1d,
                              index: Tuple[int, int]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        获取父层中的一部分权重和偏置。
        """
        weight = father_layer.weight[index[0]: index[1]].clone()
        bias = father_layer.bias[index[0]: index[1]].clone()
        return weight, bias

    def set_subset_parameters(self: Self,
                              weight: torch.Tensor,
                              bias: torch.Tensor,
                              index: Tuple[int, int]) -> None:
        """
        将权重和偏置的一部分设置到当前层中。
        """
        self.weight[index[0]: index[1]] = weight
        self.bias[index[0]: index[1]] = bias

    @staticmethod
    def parse_fraction_strings(fraction_str: str) -> Fraction:
        """
        解析字符串形式的分数并返回 Fraction 对象。
        """
        if fraction_str == '0':
            return Fraction(0, 1)
        if fraction_str == '1':
            return Fraction(1, 1)
        numerator, denominator = map(int, fraction_str.split('/'))
        return Fraction(numerator, denominator)
