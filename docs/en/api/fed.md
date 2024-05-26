# heflwr.fed
The `heflwr.fed` module provides direct support for aggregation and distribution operations in system heterogeneous federated learning.
For federated learning under system heterogeneity, direct aggregation and distribution of parameters are not feasible. `heflwr.fed` provides support for PyTorch and Flower aggregation and distribution functions.

The `heflwr.fed` module provides APIs supporting PyTorch and Flower to facilitate research on federated learning under system heterogeneity. This module provides three functions: `aggregate_layer`, `extract`, and `merge`.
- With the `aggregate_layer` function and the `reset_parameters_from_father` method of members in `heflwr.nn`, you can quickly build a simulated federated learning system implemented using PyTorch.
- With the `extract` and `merge` functions, you can quickly build a federated learning system implemented using Flower, which can be deployed in real-world application environments without the need to focus on low-level parameter transmission serialization protocols and communication details.

## aggregate_layer
> ```python
> aggregate_layer(global_layer: heflwr.nn.SUPPORT_LAYER,
>                 subset_layers: List[heflwr.nn.SUPPORT_LAYER],
>                 weights: List[int],
> ) -> None
Aggregate parameters from multiple client layers (`subset_layers`) into the global layer (`global_layer`),
where the influence of each client layer on the global layer is proportional to its associated weight.
For the definition of `SUPPORT_LAYER`, please refer to [`heflwr.nn` documentation](TODO).

### Parameters
- **global_layer** (<font color=#ED564A>_heflwr.nn.SUPPORT_LAYER_</font>) - The global parameter layer to aggregate as the target.
- **subset_layers** (<font color=#ED564A>_List[heflwr.nn.SUPPORT_LAYER]_</font>) - List of multiple local parameter layers.
- **weights** (<font color=#ED564A>_List[int]_</font>) - Weights for each client layers, typically set to a list consisting of the number of training samples for each federated client.

### Returns and Side Effects
- Returns - `None`.
- Side Effects - Parameters of the `global_layer` object are reset to the weighted average of `subset_layers`.

## extract
> ```python
> extract(parameters: flwr.common.typing.Parameters,
>         client_net: torch.nn.Module,
>         server_net: torch.nn.Module,
> ) -> flwr.common.typing.Parameters
Extract complete parameters `parameters` matching `server_net` according to the pruning structure of `client_net`,
used for the scenario where the server distributes model parameters to clients in Flower federated learning.
Returns the extracted local parameters, resets the parameters of `server_net` to `parameters`,
and resets the parameters of `client_net` to the returned values.

### Parameters
- **parameters** (<font color=#ED564A>_flwr.common.typing.Parameters_</font>) - Parameters object of the current global model.
- **client_net** (<font color=#ED564A>_torch.nn.Module_</font>) - Client model object constructed with `SUPPORT_LAYER` or a model object consistent with the client model structure.
- **server_net** (<font color=#ED564A>_torch.nn.Module_</font>) - Global model object constructed with `SUPPORT_LAYER` or a model object consistent with the global model structure.

### Returns and Side Effects
- Returns - Parameters extracted from `parameters` that match the structure of `client_net`.
- Side Effects
  - Resets the parameters of `server_net` to `parameters`.
  - Resets the parameters of `client_net` to the returned values.

## merge
> ```python
> merge(results: List[Tuple[flwr.server.client_proxy.ClientProxy,
>                           flwr.common.typing.FitRes]],
>       client_nets: List[torch.nn.Module],
>       server_net: torch.nn.Module,
> ) -> flwr.common.typing.Parameters
The parameters from models of multiple clients (contained in `results`) are aggregated using heterogeneous weighted aggregation.
used for the scenario where the server aggregates local training results from various clients in Flower federated learning.
Returns the parameters after heterogeneous weighted aggregation, resets the parameters of each object in `client_nets` to the corresponding parameters in `results`,
and resets the parameters of `server_net` to the returned values.

### Parameters
- **results** (<font color=#ED564A>_List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.typing.FitRes]]_</font>) -
Local training results of various clients.
- **client_nets** (<font color=#ED564A>_List[torch.nn.Module]_</font>) - Multiple client model objects constructed with `SUPPORT_LAYER` or model objects consistent with the client model structure.
- **server_net** (<font color=#ED564A>_torch.nn.Module_</font>) - Global model object constructed with `SUPPORT_LAYER` or a model object consistent with the global model structure.

### Returns and Side Effects
- Returns - Parameters after heterogeneous weighted aggregation.
- Side Effects
  - Resets the parameters of `server_net` to the returned values.
  - Resets the parameters of each object in `client_nets` to the corresponding parameters in `results`.
