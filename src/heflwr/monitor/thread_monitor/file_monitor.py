import os
import time
import threading
import datetime

import psutil

from ..utils.network import NetTrafficMonitor
from ..utils.power import PowerMonitor


class FileMonitor:
    def __init__(self, file, interval=5):
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.file = file
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.metrics = {
            'cpu': True,
            'memory': True,
            'network': True,
            'power': True,
        }
        self._stats = dict({
            'cpu_usage': list(),
            'memory_usage': list(),
            'network_bytes_sent': list(),
            'network_bytes_recv': list(),
            'power_vdd_in': list(),
            'power_vdd_cpu_gpu_cv': list(),
            'power_vdd_soc': list(),
        })

    def set_metrics(self, **kwargs):
        for metric, value in kwargs.items():
            if metric not in self.metrics:
                raise ValueError(f"Metric '{metric}' is not supported. Available metrics: {list(self.metrics.keys())}")
            if not isinstance(value, bool):
                raise TypeError(f"Value for metric '{metric}' must be a boolean, got {type(value)} instead.")
            self.metrics[metric] = value

    def update_metrics(self):
        with open(self.file, 'a') as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{current_time}\n")
            while self.monitoring:
                if self.metrics['cpu']:
                    cpu_usage = self.process.cpu_percent()
                    f.write(f"CPU Usage: {cpu_usage}%, ")
                    self._stats['cpu_usage'].append(cpu_usage)
                if self.metrics['memory']:
                    memory_usage = self.process.memory_percent()
                    f.write(f"Memory Usage: {memory_usage}%, ")
                    self._stats['memory_usage'].append(memory_usage)
                if self.metrics['network']:
                    bytes_sent, bytes_recv = NetTrafficMonitor().get_traffic()
                    f.write(f"Bytes Sent: {bytes_sent}, ")
                    f.write(f"Bytes Recv: {bytes_recv}, ")
                    self._stats['network_bytes_sent'].append(bytes_sent)
                    self._stats['network_bytes_recv'].append(bytes_recv)
                if self.metrics['power']:
                    vdd_in_power, vdd_cpu_gpu_cv_power, vdd_soc_power = PowerMonitor().get_power()
                    f.write(f"VDD In: {vdd_in_power}, ")
                    f.write(f"VDD Cpu_Gpu_Cv: {vdd_cpu_gpu_cv_power}, ")
                    f.write(f"VDD Soc: {vdd_soc_power}, ")
                    self._stats['power_vdd_in'].append(vdd_in_power)
                    self._stats['vdd_cpu_gpu_cv_power'].append(vdd_cpu_gpu_cv_power)
                    self._stats['power_vdd_soc'].append(vdd_soc_power)
                f.write("\n")
                f.flush()
                time.sleep(self.interval)

    def start(self):
        if not self.monitoring:
            self.monitoring = True
            # Start monitoring in a separate thread
            self.thread = threading.Thread(target=self.update_metrics)
            self.thread.daemon = True  # If user forget to use `stop`, thread will be killed automatically
            self.thread.start()

    def stop(self):
        if self.monitoring:
            self.monitoring = False  # stop the thread
            self.thread.join()

    def stats(self):
        return self._stats

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
