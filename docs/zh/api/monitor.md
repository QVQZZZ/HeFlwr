# heflwr.monitor

`heflwr.monitor` 模块为为任何 Python 程序的执行提供了资源监控的功能.
在资源受限的深度学习和物联网联邦学习的场景下, 训练设备面临着有限的计算 / 存储 / 网络 / 电池等资源,
通过 `heflwr.monitor` 模块, 训练设备的资源使用情况将得到持续追踪,
并实时性地写入本地文件系统或发送给远程的联邦学习服务器.

现有的绝大多数研究通过理论指标来衡量深度神经网络的资源消耗, 如 MACs / FLOPs / 参数量.
但这些指标通常是在理想化的条件下计算的,
不考虑实际硬件环境中的各种因素, 如内存带宽 / 存储器速度 / 并行计算能力 / 功耗限制等.
因此, 它们可能无法准确反映在特定硬件设备上的实际性能表现.

`heflwr.monitor` 模块能够在实际的运行环境中提供详细的资源使用数据, 与仅基于理论指标进行评估相比,
它能够考虑到实际硬件环境的复杂性, 提供更为准确和全面的性能分析.

`heflwr.monitor` 模块提供 `process_monitor` 和 `thread_monitor` 两个子模块,
每个子模块都提供 `FileMonitor`, `PrometheusMonitor` 以及 `RemoteMonitor` 这三种形式的监控器.
它们的组织形式可以被展示为:
```shell
├── heflwr.monitor
    ├── process_monitor
    │   ├── FileMonitor
    │   ├── PrometheusMonitor
    │   └── RemoteMonitor
    └── thread_monitor
        ├── FileMonitor
        ├── PrometheusMonitor
        └── RemoteMonitor
```
您应该根据各个监控器的特点选择想要的监控器, `process_monitor` 和 `thread_monitor` 的对比如下:
- `process_monitor` 以进程形式运行监控器, 并提供更精确的资源和性能监控.
- `thread_monitor` 以线程形式运行监控器, 控制更加简单, 额外开销更少. 但可能略微影响监控的准确性.

另外, `FileMonitor` / `PrometheusMonitor` / `RemoteMonitor` 的对比如下:
- `FileMonitor` 将资源监控信息保存在本地文件系统.
- `PrometheusMonitor` 将监控信息暴露到 HTTP 端口并接收联邦学习服务器 (或其他服务器) 上 Prometheus 程序的抓取.
- `RemoteMonitor` 将监控信息发送到联邦学习服务器 (或其他服务器).

它们各自支持的平台如下:

|             | File            | Prometheus      | Remote          |
|-------------|-----------------|-----------------|-----------------|
| **Process** | Linux / Windows | Linux           | Linux / Windows |
| **Thread**  | Linux / Windows | Linux / Windows | Linux / Windows |

如果您没有特殊的偏好, 并且只想以最简单的形式运行监控器, 则可以采用 `heflwr.monitor.process_monitor.FileMonitor`.
它将以进程形式运行监控器, 并保留监控日志到本地文件.

## 快速开始
将以下程序复制到 IDE 中, 并用任意代码替换 `your_main_logic()`.
```python
from heflwr.monitor.process_monitor import FileMonitor

# Initialize a monitor instance
monitor = FileMonitor(file='./log.txt', interval=5)

# Start - main - stop
monitor.start()  # Monitor begins continuous monitoring and writes logs to log.txt.
your_main_logic()  # Your deep learning code / federated learning code / any Python code
monitor.stop()  # Monitor stops monitoring and ceases log writing.

# Post-processing
detail = monitor.stats()
summary = monitor.summary()
print(detail)
print(summary)
```
您应该可以在控制台上观察到类似的输出:
```shell
{'cpu_usage': [0.0, 0.0], 'memory_usage': [0.2398851887620762, 0.2398851887620762], 'network_bytes_sent': [14399, 447], 'network_bytes_recv': [11201, 975], 'power_vdd_in': [], 'power_vdd_cpu_gpu_cv': [], 'power_vdd_soc': []}
{'avg_cpu_usage': 0.0, 'avg_memory_usage': 0.2398851887620762, 'total_network_bytes_sent': 14846, 'total_network_bytes_recv': 12176, 'total_power_vdd_in': 0, 'total_power_vdd_cpu_gpu_cv': 0, 'total_power_vdd_soc': 0}
```
同时程序会在运行目录下生成 `log.txt` 文件, 并在其中记录详细的监控信息.

## 导入方式
以下列出了所有监控器的导入方式:
```python
from heflwr.monitor.process_monitor import FileMonitor
from heflwr.monitor.process_monitor import PrometheusMonitor
from heflwr.monitor.process_monitor import RemoteMonitor
from heflwr.monitor.thread_monitor import FileMonitor
from heflwr.monitor.thread_monitor import PrometheusMonitor
from heflwr.monitor.thread_monitor import RemoteMonitor
```

## 初始化方式
以下展示了 `FileMonitor` / `PrometheusMonitor` / `RemoteMonitor` 的初始化方式.
- `FileMonitor`: 需要指定写入的文件 `file` (若不存在将创建), 以及监控的间隔 `interval` (单位: 秒).
  ```python
  monitor = FileMonitor(file="./log.txt", interval=5)
  ```
- `PrometheusMonitor`: 需要指定暴露的端口 `port` (默认为 8003), 以及监控的间隔 `interval` (单位: 秒).
    ```python
    monitor = PrometheusMonitor(port=8003, interval=5)
    ```
- `RemoteMonitor`: 需要指定远程服务器的地址 `host`, 监控的间隔 `interval` (单位: 秒),
日志标识符 `identifier` (若为 `None`, 则将以随机字符串代替), 远程日志格式 `simple` (默认为 `True`, 代表简洁格式).
    ```python
    monitor = RemoteMonitor(host="127.0.0.1:5000", interval=5, identifier=None, simple=True)
    ```
以上的初始化方式对于 `process_monitor` 和 `thread_monitor` 全部适用.

有关 `RemoteMonitor` 初始化参数 `identifier` 和 `simple` 的详细信息, 请查看 [`heflwr.log` 文档](TODO).

## 设置监控指标
`heflwr.monitor` 支持监控四类指标:
- CPU 使用率: 默认开启的监控指标. 统计被监控进程的 CPU 使用率占用系统总 CPU 使用率的比例 单位: %.
- 内存使用率: 默认开启的监控指标. 统计被监控进程的内存占用系统总内存的比例. 单位: %.
- 网络流量: 默认开启的监控指标. 统计系统的上行和下行流量信息. 单位: B.
- 设备功耗: 默认关闭的监控指标, 仅在支持 `tegrastats` 的设备上启用.
统计设备的输入电压, CPU / GPU / CV 核心的合计电压, 除 CPU / GPU/ CV 核心以外的 SOC 电压 (如内存和 nvdec 等特殊设备). 单位: mV.

在初始化监控器后, 可以手动加入或去除各个监控指标:
```python
monitor.set_metrics(cpu=False, memory=True, network=True, power=True)
```
上述示例等价于:
```python
monitor.set_metrics(cpu=False, power=True)
```

## 启用和关闭监控器
在需要监控的函数或代码段前后, 加入 `monitor.start()` 和 `monitor.stop()` 即可进行监控.
```python
monitor.start()
your_main_logic()
monitor.stop()
```

## 后处理
除了写入到本地文件, 暴露给 Prometheus 或发送到远程服务器外,
您还可以使用监控器的 `stats()` 和 `summary()` 方法以在后续的程序中处理或使用监控到的信息.
- `stats()` 方法提供每个抓取点 (在初始化时利用 `interval` 参数进行控制) 的监控日志.
- `summary()` 方法提供所有抓取点的平均 (如 CPU 占用, 单位: %) 或累计监控信息 (如功耗, 单位: mJ).

可以在任何时期, 包括 `stop()`方法被调用之前使用这两个方法, 只要监控器 `monitor` 已经被实例化:
```python
detail = monitor.stats()
summary = monitor.summary()
print(detail)
print(summary)
```
上述示例的输出信息类似于:
```shell
{'cpu_usage': [0.0, 0.0], 'memory_usage': [0.2398851887620762, 0.2398851887620762], 'network_bytes_sent': [14399, 447], 'network_bytes_recv': [11201, 975], 'power_vdd_in': [], 'power_vdd_cpu_gpu_cv': [], 'power_vdd_soc': []}
{'avg_cpu_usage': 0.0, 'avg_memory_usage': 0.2398851887620762, 'total_network_bytes_sent': 14846, 'total_network_bytes_recv': 12176, 'total_power_vdd_in': 0, 'total_power_vdd_cpu_gpu_cv': 0, 'total_power_vdd_soc': 0}
```
