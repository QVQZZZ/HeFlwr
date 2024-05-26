# heflwr.nn

The `heflwr.nn` module provides neural networks that have undergone [structured pruning](https://www.scaler.com/topics/pytorch/pytorch-pruning/) for federated learning, with neural network layer APIs similar to `torch.nn`.
With the `heflwr.nn` module, it is easy to create pruned networks of multiple parent networks, which can be trained on resource-constrained clients.
Additionally, the pruned networks can obtain parameters from the corresponding positions of the parent network, to support the federated learning server in distributing it to various clients.

Existing pruning schemes often use masks or set parameters to zero, such as PyTorch's official pruning scheme [torch.nn.utils.prune](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html),
which provides [unstructured pruning](https://www.scaler.com/topics/pytorch/pytorch-pruning/) for neural networks.
This theoretically prunes models but cannot truly speed up model inference/training or reduce memory consumption.

The structured pruning scheme provided by `heflwr.nn` can significantly reduce the training/inference pressure on endpoint devices, making it deployable in real IoT devices (such as Jetson Nano or Raspberry Pi).
`heflwr.nn` also provides special support for federated learning, whether you are using PyTorch for simulated federated learning or employing professional federated learning frameworks like Flower.
(This means that you don't need to download or use Flower, and you can still use `heflwr.nn` to build your heterogeneous federated learning programs)

The `heflwr.nn` module offers components such as `SSConv2d`, `SSLinear`, `SSBatchNorm2d`, which are part of the type `SUPPORT_LAYER`.
The prefix `SS` stands for `SubSet`, a term from a review paper on system heterogeneity in federated learning titled [Federated Learning for Computationally-Constrained Heterogeneous Devices: A Survey](https://arxiv.org/abs/2307.09182).
For modules in `torch.nn` that do not include explicit parameters (like `torch.nn.MaxPool2d`), `heflwr.nn` does not provide corresponding `SSLayer` implementations.
This is because, in model pruning or federated learning, support is only necessary for modules with parameters.


## Predefined
> ```python
> Interval = Tuple[str, str]
> Intervals = List[Tuple[str, str]]
> Layer_Range = Union[Interval, Intervals]
Most modules of `heflwr.nn` utilize some predefined data types, such as `Interval`, `Intervals`, `Layer_Range`.
These types are mainly used to express the range or interval of data. When structurally pruning neural networks, their numerical representations indicate the network parts to be retained.

- **Interval**: A tuple containing two string elements,
representing the start and end of an interval. The two strings respectively represent the start and end positions of the interval, such as `('0', '1/2')` representing the range from the start of the interval to its halfway point.
- **Intervals**: A list containing multiple Intervals.
Each Interval represents an independent segment. The Intervals type allows for the representation of multiple disjoint intervals,
such as `[('0'), ('1/4'), ('2/4', '1')]` representing selecting two intervals of 1/4 length each, covering a total of half the range overall, but at different positions.
- **Layer_Range**: A union type that can be an Interval or Intervals,
providing greater flexibility in defining certain ranges or intervals.

To avoid errors introduced by floating-point parameter passing, all numerical boundaries are passed in string form, which are then converted to `fractions.Fraction` objects internally in the class.
The aforementioned string types should be represented as a fraction (such as `'1/4'`), and they can also be represented as `'0'` or `'1'`.



## SSLinear
> ```python
> class heflwr.nn.SSLinear(
>     in_features: int,
>     out_features: int,
>     bias: bool = True,
>     in_features_ranges: Layer_Range = ('0', '1'), 
>     out_features_ranges: Layer_Range = ('0', '1'),
> )
Create a structurally pruned linear layer based on the provided number of features and feature ranges.
For the structurally pruned linear layer, the number of input features will be calculated based on the `in_features` variable and the `in_features_ranges` variable,
and the number of output channels will be calculated based on the `out_features` variable and the `out_features_ranges` variable.


### Parameters
- **in_features** (<font color=#ED564A>_int_</font>) - Number of input features.
- **out_features** (<font color=#ED564A>_int_</font>) - Number of output features.
- **bias** (<font color=#ED564A>_bool_</font>, _optional_) - If set to False, the layer will not learn an additional bias. Default: `True`.
- **in_features_ranges** (<font color=#ED564A>_Layer_Range_</font>, _optional_) - Range of input features to be retained. Default: `('0', '1')`.
- **out_features_ranges** (<font color=#ED564A>_Layer_Range_</font>, _optional_) - Range of output features to be retained. Default: `('0', '1')`.

### Methods
- `reset_parameters_from_father_layer(self: Self, father_layer: Union[flwr.nn.SSLinear, torch.nn.Linear]) -> None`
  - Description
    - Inherit corresponding parameters from the parent layer `father_layer` and apply them to the current layer.
    - This method allows the current layer to be initialized or updated based on the parameters of the parent layer, especially suitable for scenarios where the server distributes models to clients in model pruning or federated learning.
  - Parameters
    - `self`: Current object.
    - `father_layer`: Parent linear layer from which to inherit parameters. Can be `flwr.nn.SSLinear` or `torch.nn.Linear`.
  - Returns - `None`.

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
Create a structurally pruned convolutional layer based on the provided number of channels and channel ranges.
For the structurally pruned convolutional layer, the number of input channels will be calculated based on the `in_channels` variable and the `in_channels_ranges` variable,
and the number of output channels will be calculated based on the `out_channels` variable and the `out_channels_ranges` variable.

### Parameters
- **in_channels** (<font color=#ED564A>_int_</font>) - Number of input channels.
- **out_channels** (<font color=#ED564A>_int_</font>) - Number of output channels.
- **kernel_size** (<font color=#ED564A>_Union[int, Tuple[int, int]]_</font>) - Size of the convolution kernel.
- **stride** (<font color=#ED564A>_Union[int, Tuple[int, int]]_</font>, _optional_) - Stride of the convolution. Default: 1.
- **padding** (<font color=#ED564A>_Union[int, Tuple[int, int]]_</font>, _optional_) - Refer to the [torch.nn.Conv2d documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d).
- **dilation** (<font color=#ED564A>_Union[int, Tuple[int, int]]_</font>, _optional_) - Refer to the [torch.nn.Conv2d documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d).
- **groups** (<font color=#ED564A>_int_</font>, _optional_) - Refer to the [torch.nn.Conv2d documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d).
- **bias** (<font color=#ED564A>_bool_</font>, _optional_) - If set to False, the layer will not learn an additional bias. Default: `True`.
- **in_features_ranges** (<font color=#ED564A>_Layer_Range_</font>, _optional_) - Range of input channels to be retained. Default: `('0', '1')`.
- **out_features_ranges** (<font color=#ED564A>_Layer_Range_</font>, _optional_) - Range of output channels to be retained. Default: `('0', '1')`.

### Methods
- `reset_parameters_from_father_layer(self: Self, father_layer: Union[flwr.nn.SSConv2d, torch.nn.Conv2d]) -> None`
  - Description
    - Inherit corresponding parameters from the parent layer `father_layer` and apply them to the current layer.
    - This method allows the current layer to be initialized or updated based on the parameters of the parent layer, especially suitable for scenarios where the server distributes models to clients in model pruning or federated learning.
  - Parameters
    - `self`: Current object.
    - `father_layer`: Parent convolutional layer from which to inherit parameters. Can be `flwr.nn.SSConv2d` or `torch.nn.Conv2d`.
  - Returns - `None`.
