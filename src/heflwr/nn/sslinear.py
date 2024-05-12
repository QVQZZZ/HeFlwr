import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional
from typing_extensions import Self
from fractions import Fraction

Interval = Tuple[str, str]  # 如(0,0.5)代表提取前50%
Intervals = List[Tuple[str, str]]  # 如[(0,0.2),(0.5,0.8)]也代表提取50%，只不过提取的位置不同
Layer_Range = Union[Interval, Intervals]


class SSLinear(nn.Linear):
    def __init__(self: Self, in_features: int, out_features: int, bias: bool = True,
                 in_features_ranges: Layer_Range = ('0', '1'),
                 out_features_ranges: Layer_Range = ('0', '1')):

        # if isinstance(in_features_ranges, Interval):
        if isinstance(in_features_ranges[0], str):
            in_features_ranges = [in_features_ranges]
        # if isinstance(out_features_ranges, Interval):
        if isinstance(out_features_ranges[0], str):
            out_features_ranges = [out_features_ranges]

        in_features_ranges = [tuple(map(self.parse_fraction_strings, range_str)) for range_str in in_features_ranges]
        out_features_ranges = [tuple(map(self.parse_fraction_strings, range_str)) for range_str in out_features_ranges]

        super(SSLinear, self).__init__(in_features=sum(int(in_features * (end - start)) for start, end in in_features_ranges),
                                       out_features=sum(int(out_features * (end - start)) for start, end in out_features_ranges),
                                       bias=bias)
        # int or ceil?
        # which one is better?
        # sum(ceil(in_features * (end - start)) for start, end in in_features_range)
        # ceil(in_features * sum(end - start for start, end in in_features_range))
        self.in_features_ranges = in_features_ranges
        self.out_features_ranges = out_features_ranges
        self.reset_parameters()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self: Self) -> str:
        base_str = super().extra_repr()
        return (f'{base_str}, '
                f'in_features_ranges={self.in_features_ranges}, '
                f'out_features_ranges={self.out_features_ranges}')

    def reset_parameters_from_father_layer(self: Self, father_layer: Union[Self, nn.Linear]) -> None:
        father_out_indices_start = [int(out_range[0] * father_layer.out_features) for out_range in self.out_features_ranges]
        father_out_indices_end = [int(out_range[1] * father_layer.out_features) for out_range in self.out_features_ranges]
        father_out_indices = list(zip(father_out_indices_start, father_out_indices_end))

        father_in_indices_start = [int(in_range[0] * father_layer.in_features) for in_range in self.in_features_ranges]
        father_in_indices_end = [int(in_range[1] * father_layer.in_features) for in_range in self.in_features_ranges]
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
    def get_subset_parameters(father_layer: nn.Linear, out_index: List[Tuple[int, int]], in_index: List[Tuple[int, int]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # TODO
        weight = father_layer.weight[out_index[0]: out_index[1], in_index[0]: in_index[1]].clone()
        bias = None
        if father_layer.bias is not None:
            bias = father_layer.bias[out_index[0]: out_index[1]].clone()
        return weight, bias

    def set_subset_parameters(self: Self, weight: torch.Tensor, bias: torch.Tensor, out_index: List[Tuple[int, int]], in_index: List[Tuple[int, int]]) -> None:
        self.weight[out_index[0]: out_index[1], in_index[0]: in_index[1]] = weight
        if bias is not None:
            self.bias[out_index[0]: out_index[1]] = bias

    @staticmethod
    def parse_fraction_strings(fraction_str: str) -> Fraction:
        if fraction_str == '0':
            return Fraction(0, 1)
        if fraction_str == '1':
            return Fraction(1, 1)
        numerator, denominator = map(int, fraction_str.split('/'))
        return Fraction(numerator, denominator)


if __name__ == '__main__':
    # test_monitor example 1
    linear1 = SSLinear(in_features=20, out_features=16, bias=True)
    linear2 = SSLinear(in_features=16, out_features=8, bias=True)
    linear3 = SSLinear(in_features=8, out_features=2, bias=True)
    layers = [linear1, linear2, linear3]
    global_network = nn.Sequential(*layers)
    linear4 = SSLinear(in_features=20, out_features=16, bias=True, in_features_ranges=('0', '1'), out_features_ranges=('0', '1/2'))  # 该层的实际尺寸为20->8。该层位输入层，其in_features_range必须为(0,1)。
    linear5 = SSLinear(in_features=16, out_features=8, bias=True, in_features_ranges=('0', '1/2'), out_features_ranges=('0', '1/4'))  # 该层的实际尺寸为8->2。该层的in_features_range必须与上一层的out_features_range相等。
    linear6 = SSLinear(in_features=8, out_features=2, bias=True, in_features_ranges=('0', '1/4'), out_features_ranges=('0', '1'))  # 该层的实际尺寸为2->2。该层的in_features_range必须与上一层的out_features_range相等。该层位输出层，其out__features_range必须为(0,1)。
    layers = [linear4, linear5, linear6]
    subset_network = nn.Sequential(*layers)
    x = torch.rand([1, 20])
    print(global_network(x))
    print(subset_network(x))

    # test_monitor example 2
    linear1 = SSLinear(in_features=5, out_features=3, bias=True)
    linear2 = SSLinear(in_features=5, out_features=3, bias=True,
                       in_features_ranges=[('0', '2/5'), ('3/5', '4/5')],
                       out_features_ranges=[('0', '1/3'), ('2/3', '1')])

    x = torch.rand([1, 5])
    print(linear1(x))
    x = torch.rand([1, 3])
    print(linear2(x))

    # test_monitor example 3
    linear1 = SSLinear(in_features=6, out_features=4, bias=True)
    linear2 = SSLinear(in_features=6, out_features=4, bias=True,
                       in_features_ranges=[('0', '2/6'), ('3/6', '4/6'), ('5/6', '1')],
                       out_features_ranges=[('0', '2/4'), ('3/4', '1')],
                       father_layer=linear1)
    x = torch.rand([1, 6])
    print(linear1(x))
    x = torch.rand([1, 4])
    print(linear2(x))

    # test_monitor example 4
    linear1 = SSLinear(in_features=20, out_features=16, bias=True)
    relu1 = nn.ReLU()
    linear2 = SSLinear(in_features=16, out_features=8, bias=True)
    relu2 = nn.ReLU()
    linear3 = SSLinear(in_features=8, out_features=2, bias=True)
    layers = [linear1, relu1, linear2, relu2, linear3]
    global_network = nn.Sequential(*layers)

    linear4 = SSLinear(in_features=20, out_features=16, bias=True, in_features_ranges=('0', '1'), out_features_ranges=('0', '1/2'), father_layer=linear1)  # 该层的实际尺寸为20->8。该层位输入层，其in_features_range必须为(0,1)。
    relu3 = nn.ReLU()
    linear5 = SSLinear(in_features=16, out_features=8, bias=True, in_features_ranges=('0', '1/2'), out_features_ranges=('0', '1/4'), father_layer=linear2)  # 该层的实际尺寸为8->2。该层的in_features_range必须与上一层的out_features_range相等。
    relu4 = nn.ReLU()
    linear6 = SSLinear(in_features=8, out_features=2, bias=True, in_features_ranges=('0', '1/4'), out_features_ranges=('0', '1'), father_layer=linear3)  # 该层的实际尺寸为2->2。该层的in_features_range必须与上一层的out_features_range相等。该层位输出层，其out__features_range必须为(0,1)。
    layers = [linear4, relu3, linear5, relu4, linear6]
    subset_network = nn.Sequential(*layers)

    x = torch.rand([1, 20])
    print(global_network(x))
    print(subset_network(x))

    # test_monitor example 5
    linear1 = SSLinear(in_features=5, out_features=4, bias=True)
    linear2 = SSLinear(in_features=4, out_features=2, bias=True)

    linear3 = SSLinear(in_features=5, out_features=4, bias=True, in_features_ranges=('0', '1'), out_features_ranges=('0', '1/2'), father_layer=linear1)
    linear4 = SSLinear(in_features=4, out_features=2, bias=True, in_features_ranges=('0', '1/2'), out_features_ranges=('0', '1'), father_layer=linear2)
    linear4_ = SSLinear(in_features=4, out_features=2, bias=True, in_features_ranges=('1/2', '1'), out_features_ranges=('0', '1'), father_layer=linear2)

    layers = [linear3, linear4]
    subnet1 = nn.Sequential(*layers)
    layers = [linear3, linear4_]
    subnet2 = nn.Sequential(*layers)

    x = torch.rand([1, 5])
    print(subnet1(x))
    print(subnet2(x))

    # test_monitor example 6: aggregate
    from aggregate import aggregate_layer

    global_linear = SSLinear(in_features=6, out_features=4, bias=True)
    subset_linear1 = SSLinear(in_features=6, out_features=4, bias=True, in_features_ranges=('0', '1'), out_features_ranges=('0', '1'))
    subset_linear2 = SSLinear(in_features=6, out_features=4, bias=True, in_features_ranges=('0', '1/2'), out_features_ranges=('0', '1/2'))
    subset_linear3 = SSLinear(in_features=6, out_features=4, bias=True, in_features_ranges=('0', '1/2'), out_features_ranges=('0', '1/2'))
    subset_linear4 = SSLinear(in_features=6, out_features=4, bias=True, in_features_ranges=[('0', '1/6'), ('1/2', '1')], out_features_ranges=('0', '1'), father_layer=global_linear)
    sub_layers = [subset_linear1, subset_linear2, subset_linear3, subset_linear4]
    with torch.no_grad():
        for i, layer in enumerate(sub_layers, 1):
            layer.weight.data.fill_(i)
            layer.bias.data.fill_(i)
    samples = [10, 20, 30, 40]
    aggregate_layer(global_linear, sub_layers, weights=samples)
