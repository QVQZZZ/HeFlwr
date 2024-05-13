import time

import psutil


class NetTrafficMonitor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(NetTrafficMonitor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.last_sent, self.last_recv = self.get_net_io()
            self.first_call = True
            self.initialized = True

    @staticmethod
    def get_net_io():
        net_io = psutil.net_io_counters()
        return net_io.bytes_sent, net_io.bytes_recv

    def get_traffic(self):
        if self.first_call:
            time.sleep(5)  # Wait for a second to get initial reading
            self.first_call = False
        curr_sent, curr_recv = self.get_net_io()
        diff_sent, diff_recv = curr_sent - self.last_sent, curr_recv - self.last_recv
        self.last_sent, self.last_recv = curr_sent, curr_recv
        return diff_sent, diff_recv
