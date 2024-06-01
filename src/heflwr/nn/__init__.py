from typing import Union, List, Tuple

from .sslinear import SSLinear
from .ssconv2d import SSConv2d
from .ssbatchnorm2d import SSBatchNorm2d


SUPPORT_LAYER = Union[
    SSLinear,
    SSConv2d,
    SSBatchNorm2d
]

# For example, (0, 0.5) represents extracting the first 50%.
Interval = Tuple[str, str]
# For example, [(0, 0.2), (0.5, 0.8)] also represents extracting 50%, but at different positions.
Intervals = List[Tuple[str, str]]
Layer_Range = Union[Interval, Intervals]
