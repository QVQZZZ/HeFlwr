import subprocess
import re


def get_jetson_model():
    try:
        model = subprocess.check_output('cat /sys/firmware/devicetree/base/model', shell=True)
        model = str(model, encoding='utf-8')  # model.decode('utf-8')
        return model.strip()
    except Exception as ex:
        return str(ex)


class PowerMonitor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PowerMonitor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.e = 0.3  # Toleration to execute the cmd, different jetson model may need different e
            self.cmd = ['sudo', 'tegrastats']
            self.initialized = True
            self.jetson_model = get_jetson_model()
            self.patterns = {
                'Jetson-AGX\x00': r'(GPU|CPU|SOC|CV|VDDRQ|SYS5V) (\d+)mW',
                'NVIDIA Orin Nano Developer Kit\x00': r'(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC) (\d+)mW',
            }

    def _extract_power_dict(self, first_line):
        if self.jetson_model in self.patterns:
            pattern = self.patterns[self.jetson_model]
            matches = re.findall(pattern, first_line)
            power_data = {}
            for component, power in matches:
                power_data[component] = int(power)
            return power_data
        else:
            raise RuntimeError("Unsupported jetson device type")

    def get_power(self):
        try:
            # text=True may have no effect, cause stdout is returned by exception but not normal return
            # See https://www.datacamp.com/tutorial/python-subprocess
            subprocess.run(self.cmd, capture_output=True, text=True, timeout=1+self.e)
        except subprocess.TimeoutExpired as ex:
            if ex.stdout:
                first_line = ex.stdout.splitlines()[0]
            else:
                raise RuntimeError(f"No output was captured. Current e={self.e}, try a large number of 'e'.")
            if isinstance(first_line, bytes):  # text=True may have no effect, so we need to confirm `first_line` is a str
                first_line = str(first_line, encoding='utf-8')
                power_data = self._extract_power_dict(first_line)
                """
                `Jetson stats` for jetson devices.
                See https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html
                Orin Series: https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#jetson-orin-nx-series-and-jetson-orin-nano-series
                Xavier Series: https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonXavierNxSeriesAndJetsonAgxXavierSeries.html#sd-platformpowerandperformance-jetsonxaviernxandjetsonagxxavier
                """
                if self.jetson_model in ('Jetson-AGX\x00', ):
                    # power_data = {
                    #     'GPU': 0,
                    #     'CPU': 0,
                    #     'SOC': 0,
                    #     'CV': 0,
                    #     'VDDRQ': 0,
                    #     'SYS5V': 0
                    # }
                    vdd_in_power = sum(power_data.values())
                    vdd_cpu_gpu_cv_power = power_data['CPU'] + power_data['GPU'] + power_data['CV']
                    vdd_soc_power = power_data['SOC']
                elif self.jetson_model in ('NVIDIA Orin Nano Developer Kit\x00', ):
                    # power_data = {
                    #     'VDD_IN': 0,
                    #     'VDD_CPU_GPU_CV': 0,
                    #     'VDD_SOC': 0
                    # }
                    vdd_in_power, vdd_cpu_gpu_cv_power, vdd_soc_power = power_data.values()
                else:
                    raise RuntimeError("Unsupported jetson device type")
                return vdd_in_power, vdd_cpu_gpu_cv_power, vdd_soc_power
        except Exception as ex:
            raise RuntimeError(f"subprocess.run encountered an error during execution '{' '.join(self.cmd)}'.\n"
                               f"Origin Error: {ex}")


# if __name__ == '__main__':
#     print(PowerMonitor().get_power())
