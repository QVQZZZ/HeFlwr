# heflwr.fed
`heflwr.fed` 模块为系统异构联邦学习中的聚合和分发操作提供直接支持.
对于系统异构性下的联邦学习, 不能直接对参数进行聚合和分发, `heflwr.fed` 提供支持 PyTorch
和 Flower 的聚合和分发函数.

`heflwr.fed` 模块提供了支持 PyTorch 和 Flower 的 APIs,
以支持对系统异构性下的联邦学习研究. 该模块提供 `aggregate_layer`, `extract`, `merge` 三个函数.
- 借助 `aggregate_layer` 函数以及
`fed.nn` 中成员的 `reset_parameters_from_father` 方法,
可以快速构建一个利用 PyTorch 实现的模拟联邦学习系统.
- 借助 `extract` 以及 `merge` 函数, 可以快速构建一个利用 Flower 实现的,
可以部署于真实应用环境下的联邦学习系统, 而不必关注底层参数传输的序列化协议以及通信细节.

## aggregate_layer
> ```python
> aggregate_layer(global_layer: heflwr.nn.SUPPORT_LAYER,
>                 subset_layers: List[heflwr.nn.SUPPORT_LAYER],
>                 weights: List[int],
> ) -> None
将来自多个子层 (`subset_layers`) 的参数聚合到全局层 (`global_layer`) 中,
其中每个子层对全局层的影响与其关联的权重成正比.
`SUPPORT_LAYER` 的定义, 请参阅 [`heflwr.nn` 文档](TODO).

### 参数
- **global_layer** (<font color=#ED564A>_heflwr.nn.SUPPORT_LAYER_</font>) - 作为聚合目标的全局参数层.
- **subset_layers** (<font color=#ED564A>_List[heflwr.nn.SUPPORT_LAYER]_</font>) - 多个局部参数层组成的列表.
- **weights** (<font color=#ED564A>_List[int]_</font>) - 每个子层的权重, 通常设置为每个联邦客户端训练样本数组成的列表.

### 返回值和副作用
- 返回值 - `None`.
- 副作用 - `global_layer` 对象的参数被重新设置为 `subset_layers` 的加权平均.

## extract
> ```python
> extract(parameters: flwr.common.typing.Parameters,
>         client_net: torch.nn.Module,
>         server_net: torch.nn.Module,
> ) -> flwr.common.typing.Parameters
将匹配 `server_net` 的完整参数 `parameters` 按照 `client_net` 的剪枝结构进行提取,
用于 Flower 联邦学习中服务器向客户端分发模型参数的情形.
返回提取后的局部参数, 并将 `server_net` 的参数重新设置为 `parameters`,
将 `client_net` 的参数重新设置为返回值.

### 参数
- **parameters** (<font color=#ED564A>_flwr.common.typing.Parameters_</font>) -.
当前全局模型的参数对象.
- **client_net** (<font color=#ED564A>_torch.nn.Module_</font>) -
以 `SUPPORT_LAYER` 构建的客户端模型对象或与客户端模型结构一致的模型对象.
- **server_net** (<font color=#ED564A>_torch.nn.Module_</font>) -
以 `SUPPORT_LAYER` 构建的全局模型对象或与全局模型结构一致的模型对象.

### 返回值和副作用
- 返回值 - 从 `parameters` 中提取出的, 符合 `client_net` 结构的参数.
- 副作用
  - 将 `server_net` 的参数重新设置为 `parameters`.
  - 将 `client_net` 的参数重新设置为返回值.

## merge

> ```python
> merge(results: List[Tuple[flwr.server.client_proxy.ClientProxy,
>                           flwr.common.typing.FitRes]],
>       client_nets: List[torch.nn.Module],
>       server_net: torch.nn.Module,
> ) -> flwr.common.typing.Parameters
将来自多个客户端模型的参数 (包含在 `results` 中) 进行异构加权聚合,
用于 Flower 联邦学习中服务器聚合各个客户端本地训练结果的情形.
返回异构加权聚合后的参数, 并将 `client_nets` 中的每个对象的参数重新设置为对应 `results` 中的参数,
将 `server_net` 的参数重新设置为返回值.

### 参数
- **results** (<font color=#ED564A>_List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.typing.FitRes]]_</font>) -
各个客户端的本地训练结果.
- **client_nets** (<font color=#ED564A>_List[torch.nn.Module]_</font>) -
以 `SUPPORT_LAYER` 构建的多个客户端模型对象或与客户端模型结构一致的模型对象.
- **server_net** (<font color=#ED564A>_torch.nn.Module_</font>) -
以 `SUPPORT_LAYER` 构建的全局模型对象或与全局模型结构一致的模型对象.

### 返回值和副作用
- 返回值 - 经过异构加权聚合后的参数.
- 副作用
  - 将 `server_net` 的参数重新设置为返回值.
  - 将 `client_nets` 中的每个对象的参数重新设置为对应对应 `results` 中的参数.
