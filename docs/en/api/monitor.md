# heflwr.monitor

The `heflwr.monitor` module provides functionality for resource monitoring for the execution of any Python program.
In scenarios like resource-constrained deep learning and Internet of Things (IoT) federated learning, training devices face limited resources such as computation, storage, network, and battery.
With the `heflwr.monitor` module, the resource usage of training devices will be continuously tracked,
and written to the local file system or sent to a remote federated learning server in real-time.

Most existing research measures the resource consumption of deep neural networks using theoretical metrics such as MACs (Multiply-Accumulate Operations), FLOPs (Floating Point Operations), and parameters.
However, these metrics are usually calculated under idealized conditions,
neglecting various factors in the actual hardware environment, such as memory bandwidth, storage speed, parallel computing capability, and power constraints.
Therefore, they may not accurately reflect the actual performance on specific hardware devices.

The `heflwr.monitor` module can provide detailed resource usage data in the actual operating environment.
Compared to evaluation based solely on theoretical metrics,
it can take into account the complexity of the actual hardware environment, providing more accurate and comprehensive performance analysis.

The `heflwr.monitor` module provides two sub-modules: `process_monitor` and `thread_monitor`.
Each sub-module offers three forms of monitors: `FileMonitor`, `PrometheusMonitor`, and `RemoteMonitor`.
Their organizational structure can be illustrated as follows:
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
Please choose the monitor you want based on the characteristics of each monitor. The comparison between `process_monitor` and `thread_monitor` is as follows:
- `process_monitor` runs the monitor in the form of a process, providing more accurate resource and performance monitoring.
- `thread_monitor` runs the monitor in the form of a thread, offering simpler control and less additional overhead. However, it may slightly affect the accuracy of monitoring.

Furthermore, the comparison between `FileMonitor` / `PrometheusMonitor` / `RemoteMonitor` is as follows:
- `FileMonitor` stores resource monitoring information in the local file system.
- `PrometheusMonitor` exposes monitoring information to an HTTP port and receives scraping by Prometheus running on the federated learning server (or other servers).
- `RemoteMonitor` sends monitoring information to the federated learning server (or other servers).

Their platform support is as follows:

|             | File            | Prometheus      | Remote          |
|-------------|-----------------|-----------------|-----------------|
| **Process** | Linux / Windows | Linux           | Linux / Windows |
| **Thread**  | Linux / Windows | Linux / Windows | Linux / Windows |

If you have no specific preference and just want to run the monitor in the simplest form, you can use `heflwr.monitor.process_monitor.FileMonitor`. It runs the monitor in the form of a process and logs monitoring data to a local file.


## Quick Start
Copy the following code into your IDE, and replace `your_main_logic()` with your actual code.
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
You should be able to observe similar output on the console:
```shell
{'cpu_usage': [0.0, 0.0], 'memory_usage': [0.2398851887620762, 0.2398851887620762], 'network_bytes_sent': [14399, 447], 'network_bytes_recv': [11201, 975], 'power_vdd_in': [], 'power_vdd_cpu_gpu_cv': [], 'power_vdd_soc': []}
{'avg_cpu_usage': 0.0, 'avg_memory_usage': 0.2398851887620762, 'total_network_bytes_sent': 14846, 'total_network_bytes_recv': 12176, 'total_power_vdd_in': 0, 'total_power_vdd_cpu_gpu_cv': 0, 'total_power_vdd_soc': 0}
```
At the same time, the program will generate a `log.txt` file in the running directory, recording detailed monitoring information therein.

## Import
Below are the import methods for all monitors:
```python
from heflwr.monitor.process_monitor import FileMonitor
from heflwr.monitor.process_monitor import PrometheusMonitor
from heflwr.monitor.process_monitor import RemoteMonitor
from heflwr.monitor.thread_monitor import FileMonitor
from heflwr.monitor.thread_monitor import PrometheusMonitor
from heflwr.monitor.thread_monitor import RemoteMonitor
```

## Initialization
Below are the initialization methods for `FileMonitor`, `PrometheusMonitor`, and `RemoteMonitor`.
- `FileMonitor`: You need to specify the file to write to `file` (if it does not exist, it will be created), and the monitoring interval `interval` (unit: seconds).
  ```python
  monitor = FileMonitor(file="./log.txt", interval=5)
  ```
- `PrometheusMonitor`: You need to specify the port to expose `port` (default is 8003), and the monitoring interval `interval` (unit: seconds).
    ```python
    monitor = PrometheusMonitor(port=8003, interval=5)
    ```
- `RemoteMonitor`: You need to specify the address of the remote server host, the monitoring interval `interval` (unit: seconds),
log identifier `identifier` (if `None`, a random string will be used instead), and the remote log format `simple` (default is `True`, representing concise format).
    ```python
    monitor = RemoteMonitor(host="127.0.0.1:5000", interval=5, identifier=None, simple=True)
    ```
The above initialization methods apply to both `process_monitor` and `thread_monitor`.
For detailed information about the initialization parameters `identifier` and `simple` for `RemoteMonitor`, please refer to the [`heflwr.log` documentation](TODO).


## Setting Monitoring Metrics
`heflwr.monitor` supports monitoring four types of metrics:
- CPU Usage: Monitoring metric enabled by default. It calculates the percentage of CPU usage of the monitored process compared to the total CPU usage of the system. Unit: %.
- Memory Usage: Monitoring metric enabled by default. It calculates the percentage of memory usage of the monitored process compared to the total memory of the system. Unit: %.
- Network Traffic: Monitoring metric enabled by default. It monitors the system's upstream and downstream traffic information. Unit: B.
- Device Power Consumption: Monitoring metric disabled by default, only enabled on devices that support `tegrastats`.
It monitors the input voltage of the device, the total voltage of CPU / GPU / CV cores, and the SOC voltage excluding CPU / GPU/ CV cores (such as memory and nvdec). Unit: mV.

After initializing the monitor, you can manually add or remove each monitoring metric:

```python
monitor.set_metrics(cpu=False, memory=True, network=True, power=True)
```
The above example is equivalent to:
```python
monitor.set_metrics(cpu=False, power=True)
```

## Start and Stop the Monitor
Add `monitor.start()` and `monitor.stop()` before and after the function or code segment that needs monitoring.
```python
monitor.start()
your_main_logic()
monitor.stop()
```

## Post-processing
In addition to writing to a local file, exposing to Prometheus, or sending to a remote server,
you can also use the monitor's `stats()` and `summary()` methods to process or use the monitored information in subsequent programs.
- The `stats()` method provides monitoring logs for each sampling point (controlled by the `interval` parameter during initialization).
- The `summary()` method provides average (e.g., CPU usage, unit: %) or cumulative monitoring information (e.g., power consumption, unit: mJ) for all sampling points.

You can use these two methods at any time, including before the `stop()` method is called, as long as the monitor `monitor` has been instantiated.

```python
detail = monitor.stats()
summary = monitor.summary()
print(detail)
print(summary)
```
The output information of the above example is similar to:
```shell
{'cpu_usage': [0.0, 0.0], 'memory_usage': [0.2398851887620762, 0.2398851887620762], 'network_bytes_sent': [14399, 447], 'network_bytes_recv': [11201, 975], 'power_vdd_in': [], 'power_vdd_cpu_gpu_cv': [], 'power_vdd_soc': []}
{'avg_cpu_usage': 0.0, 'avg_memory_usage': 0.2398851887620762, 'total_network_bytes_sent': 14846, 'total_network_bytes_recv': 12176, 'total_power_vdd_in': 0, 'total_power_vdd_cpu_gpu_cv': 0, 'total_power_vdd_soc': 0}
```
