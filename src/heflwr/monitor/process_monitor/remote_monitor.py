import os
import time
import multiprocessing

import psutil
import uuid

from ...log.logger import logger, configure
from ..utils.network import NetTrafficMonitor
from ..utils.power import PowerMonitor


class RemoteMonitor:
    def __init__(self, host, interval=5, identifier=None, simple=True):
        self.pid = os.getpid()
        # self.process = psutil.Process(self.pid)
        self.host = host
        self.interval = interval
        self.monitoring = multiprocessing.Event()
        self.monitor_process = None
        self.metrics = {
            'cpu': True,
            'memory': True,
            'network': True,
            'power': True,
        }
        manager = multiprocessing.Manager()
        self._stats = manager.dict({
            'cpu_usage': manager.list(),
            'memory_usage': manager.list(),
            'network_bytes_sent': manager.list(),
            'network_bytes_recv': manager.list(),
            'power_vdd_in': manager.list(),
            'power_vdd_cpu_gpu_cv': manager.list(),
            'power_vdd_soc': manager.list(),
        })

        self.identifier = f"remote-monitor-{uuid.uuid4().hex}" if identifier is None else identifier
        self.simple = simple
        # Main process use function `configure` can't propagate to child processes
        # configure(identifier=self.identifier, host=self.host, simple=self.simple)

    def set_metrics(self, **kwargs):
        for metric, value in kwargs.items():
            if metric not in self.metrics:
                raise ValueError(f"Metric '{metric}' is not supported. Available metrics: {list(self.metrics.keys())}")
            if not isinstance(value, bool):
                raise TypeError(f"Value for metric '{metric}' must be a boolean, got {type(value)} instead.")
            self.metrics[metric] = value

    def update_metrics(self):
        configure(identifier=self.identifier, host=self.host, simple=self.simple)
        process = psutil.Process(self.pid)
        while self.monitoring.is_set():
            message = str()
            if self.metrics['cpu']:
                cpu_usage = process.cpu_percent()
                message += f"Cpu Usage: {cpu_usage}%, "
                self._stats['cpu_usage'].append(cpu_usage)
            if self.metrics['memory']:
                memory_usage = process.memory_percent()
                message += f"Memory Usage: {memory_usage}%, "
                self._stats['memory_usage'].append(memory_usage)
            if self.metrics['network']:
                bytes_sent, bytes_recv = NetTrafficMonitor().get_traffic()
                message += f"Bytes Sent: {bytes_sent}, Bytes Recv: {bytes_recv}, "
                self._stats['network_bytes_sent'].append(bytes_sent)
                self._stats['network_bytes_recv'].append(bytes_recv)
            if self.metrics['power']:
                vdd_in_power, vdd_cpu_gpu_cv_power, vdd_soc_power = PowerMonitor().get_power()
                message += f"VDD In: {vdd_in_power}, VDD cpu_gpu_cv: {vdd_cpu_gpu_cv_power}, VDD soc: {vdd_soc_power}, "
                self._stats['power_vdd_in'].append(vdd_in_power)
                self._stats['vdd_cpu_gpu_cv_power'].append(vdd_cpu_gpu_cv_power)
                self._stats['power_vdd_soc'].append(vdd_soc_power)
            logger.info(message)
            time.sleep(self.interval)

    def start(self):
        if not self.monitoring.is_set():
            self.monitoring.set()
            self.monitor_process = multiprocessing.Process(target=self.update_metrics)
            self.monitor_process.daemon = True  # If user forget to use `stop`, thread will be killed automatically
            self.monitor_process.start()

    def stop(self):
        if self.monitoring.is_set():
            self.monitoring.clear()
            self.monitor_process.join()

    def stats(self):
        stats_dict = dict(self._stats)
        for key in stats_dict:
            stats_dict[key] = list(stats_dict[key])
        return stats_dict

    def summary(self):
        avg_cpu_usage = sum(self._stats['cpu_usage']) / len(self._stats['cpu_usage']) if self._stats['cpu_usage'] else 0
        avg_memory_usage = sum(self._stats['memory_usage']) / len(self._stats['memory_usage']) if self._stats['memory_usage'] else 0
        total_network_bytes_sent = sum(self._stats['network_bytes_sent'])
        total_network_bytes_recv = sum(self._stats['network_bytes_recv'])
        total_power_vdd_in = sum([instantaneous_power * self.interval for instantaneous_power in self._stats['power_vdd_in']])
        total_power_vdd_cpu_gpu_cv = sum([instantaneous_power * self.interval for instantaneous_power in self._stats['power_vdd_cpu_gpu_cv']])
        total_power_vdd_soc = sum([instantaneous_power * self.interval for instantaneous_power in self._stats['power_vdd_soc']])
        return {
            'avg_cpu_usage': avg_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            'total_network_bytes_sent': total_network_bytes_sent,
            'total_network_bytes_recv': total_network_bytes_recv,
            'total_power_vdd_in': total_power_vdd_in,
            'total_power_vdd_cpu_gpu_cv': total_power_vdd_cpu_gpu_cv,
            'total_power_vdd_soc': total_power_vdd_soc
        }