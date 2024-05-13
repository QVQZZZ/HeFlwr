from typing import Union

from .sslinear import SSLinear
from .ssconv2d import SSConv2d


SUPPORT_LAYER = Union[
    SSLinear,
    SSConv2d,
]
