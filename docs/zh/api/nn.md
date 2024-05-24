# heflwr.nn

`heflwr.nn` 模块为联邦学习提供经过[结构化剪枝](https://www.scaler.com/topics/pytorch/pytorch-pruning/)的神经网络, 并采用与 `torch.nn` 相似的神经网络层 API.
通过 `heflwr.nn` 模块, 可以轻松地创建多个父网络的剪枝网络, 这些剪枝网络可以在资源受限的客户端上进行训练.
同时, 剪枝后的网络可以从父网络的对应位置获取参数, 以支持联邦学习服务器向各个客户端分发它.

现有的剪枝方案通常采用掩码或将参数置 0 来实现, 如 PyTorch 的官方剪枝方案 [torch.nn.utils.prune](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html),
它为神经网络提供[非结构化剪枝](https://www.scaler.com/topics/pytorch/pytorch-pruning/),
从理论层面对模型进行剪枝, 但无法真正地加快模型的推理/训练速度或减少内存消耗.

`heflwr.nn` 提供的结构化剪枝方案可以显著降低端侧设备的训练/推理压力, 以部署在真实的物联网设备中 (如 Jetson Nano 或 Raspberry Pi).
同时 `heflwr.nn` 还为联邦学习提供了特殊的支持, 无论你使用的是 PyTorch 进行模拟联邦学习, 还是采用 Flower 等专业的联邦学习框架.
(这意味着你不需要下载或使用 Flower, 也仍然可以使用 `heflwr.nn` 来构建你的异构联邦学习程序)

`heflwr.nn` 模块提供 `SSConv2d`, `SSLinear`, `SSBatchNorm2d` 等模块.
其中前缀 `SS` 是术语 `SubSet` 的缩写, 该术语来自一篇研究联邦学习中系统异构型的综述论文 [Federated Learning for Computationally-Constrained
 Heterogeneous Devices: A Survey](https://arxiv.org/abs/2307.09182).
对于 `torch.nn` 中不包含显式参数的模块 (如`torch.nn.MaxPool2d`), `heflwr.nn` 不提供对应的 `SSLayer` 实现.
这是因为在模型剪枝或联邦学习中, 都只需要对带参数模块提供对应的支持.


## 预先定义
> ```python
> Interval = Tuple[str, str]
> Intervals = List[Tuple[str, str]]
> Layer_Range = Union[Interval, Intervals]
`heflwr.nn` 的大多数模块利用了一些预定义的数据类型, 如 `Interval`, `Intervals`, `Layer_Range`.
这些类型主要用于表达数据的范围或区间. 在对神经网络进行结构化剪枝时, 它们的数值表示需要保留的网络部分.

- **Interval**: 一个包含两个字符串元素的元组,
表示一个区间的开始和结束. 两个字符串分别表示区间的开始和结束位置, 如 `('0', '1/2')` 表示从区间开始到其一半的范围.
- **Intervals**: 一个包含多个 Interval 的列表.
每个 Interval 表示一段独立的区间, Intervals 类型允许表示多个不连续的区间,
如 `[('0'), ('1/4'), ('2/4', '1')]` 表示选取两个 1/4 长度范围的区间, 总共覆盖了整体一半的范围, 但位于不同的位置.
- **Layer_Range**: 一个联合类型, 它可以是一个 Interval 或 Intervals,
这使得在定义某些范围或者区间时提供了更大的灵活性.

为避免浮点数传参时引入的误差, 所有的数值边界均采用字符串形式传入, 它们会在类的内部转化为 `fractions.Fraction` 对象.
上述的字符串类型应该被表示为一个分数 (如 `'1/4'`), 此外它还可以被表示为 `'0'` 或 `'1'`.



## SSLinear
> ```python
> class heflwr.nn.SSLinear(
>     in_features: int,
>     out_features: int,
>     bias: bool = True,
>     in_features_ranges: Layer_Range = ('0', '1'), 
>     out_features_ranges: Layer_Range = ('0', '1'),
> )
根据提供的特征数量以及特征范围创建一个结构化剪枝后的线性层.
结构化剪枝后的线性层, 其输入特征数将通过 `in_features` 变量和 `in_features_ranges` 变量计算得出,
输出通道数将通过 `out_features` 变量和 `out_features_ranges` 变量计算得出.

### 参数
- **in_features** (<font color=#ED564A>_int_</font>) - 输入特征的数量
- **out_features** (<font color=#ED564A>_int_</font>) - 输出特征的数量
- **bias** (<font color=#ED564A>_bool_</font>, _optional_) - 如果设置为 False, 该层将不会学习附加偏差. 默认值: `True`
- **in_features_ranges** (<font color=#ED564A>_Layer_Range_</font>, _optional_) - 需要保留的输入特征的范围. 默认值: `('0', '1')`
- **out_features_ranges** (<font color=#ED564A>_Layer_Range_</font>, _optional_) - 需要保留的输出特征的范围. 默认值: `('0', '1')`

### 方法
- `reset_parameters_from_father_layer(self: Self, father_layer: Union[flwr.nn.SSLinear, torch.nn.Linear]) -> None`
  - 描述
    - 从父层 `father_layer` 继承对应位置的参数并应用于当前层.
    - 这个方法允许当前层根据父层的参数进行初始化或更新, 特别适用于模型剪枝或联邦学习中服务器为客户端分发模型的场景.
  - 参数
    - `self`: 当前对象. 
    - `father_layer`: 要继承参数的父线性层. 可以是 `flwr.nn.SSLinear` 或 `torch.nn.Linear`.
  - 返回
    - `None`

## SSConv2d
> ```python
> class heflwr.nn.SSConv2d(
>     in_channels: int,
>     out_channels: int,
>     kernel_size:  Union[int, Tuple[int, int]],
>     stride: Union[int, Tuple[int, int]] = 1,
>     padding: Union[int, Tuple[int, int]] = 0,
>     dilation: Union[int, Tuple[int, int]] = 1,
>     groups: int = 1,
>     bias: bool = True,
>     in_channels_ranges: Layer_Range = ('0', '1'), 
>     out_channels_ranges: Layer_Range = ('0', '1'),
> )
根据提供的通道数量以及特征范围创建一个结构化剪枝后的卷积层.
结构化剪枝后的卷积层, 其输入通道数将通过 `in_channels` 变量和 `in_channels_ranges` 变量计算得出,
输出通道数将通过 `out_channels` 变量和 `out_channels_ranges` 变量计算得出.

### 参数
- **in_channels** (<font color=#ED564A>_int_</font>) - 输入通道的数量
- **out_channels** (<font color=#ED564A>_int_</font>) - 输出通道的数量
- **kernel_size** (<font color=#ED564A>_Union[int, Tuple[int, int]]_</font>) - 卷积核的尺寸
- **stride** (<font color=#ED564A>_Union[int, Tuple[int, int]]_</font>, _optional_) - 卷积的步长. 默认值: 1
- **padding** (<font color=#ED564A>_Union[int, Tuple[int, int]]_</font>, _optional_) - 请参考 [torch.nn.Conv2d文档](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
- **dilation** (<font color=#ED564A>_Union[int, Tuple[int, int]]_</font>, _optional_) - 请参考 [torch.nn.Conv2d文档](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
- **groups** (<font color=#ED564A>_int_</font>, _optional_) - 请参考 [torch.nn.Conv2d文档](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
- **bias** (<font color=#ED564A>_bool_</font>, _optional_) - 如果设置为 False, 该层将不会学习附加偏差. 默认值: `True`
- **in_features_ranges** (<font color=#ED564A>_Layer_Range_</font>, _optional_) - 需要保留的输入通道的范围. 默认值: `('0', '1')`
- **out_features_ranges** (<font color=#ED564A>_Layer_Range_</font>, _optional_) - 需要保留的输出通道的范围. 默认值: `('0', '1')`

### 方法
- `reset_parameters_from_father_layer(self: Self, father_layer: Union[flwr.nn.SSConv2d, torch.nn.Conv2d]) -> None`
  - 描述
    - 从父层 `father_layer` 继承对应位置的参数并应用于当前层.
    - 这个方法允许当前层根据父层的参数进行初始化或更新, 特别适用于模型剪枝或联邦学习中服务器为客户端分发模型的场景.
  - 参数
    - `self`: 当前对象.
    - `father_layer`: 要继承参数的父卷积层. 可以是 `flwr.nn.SSConv2d` 或 `torch.nn.Conv2d`.
  - 返回
    - `None`