import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, keep_rate, training) -> None:
        super().__init__()
        if training:
            self.keep_rate = keep_rate
        else:
            self.keep_rate = 1

    def forward(self, x):
        """ Scaler forward. """
        output = x / self.keep_rate if self.training else x
        return output
