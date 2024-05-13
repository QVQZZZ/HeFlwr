import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, rate, scale):
        super().__init__()
        if scale:
            self.rate = rate
        else:
            self.rate = 1

    def forward(self, x):
        """ Scaler forward. """
        output = x / self.rate if self.training else x
        return output
