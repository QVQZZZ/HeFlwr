from typing import Union

from .sslinear import SSLinear
from .ssconv2d import SSConv2d
from .ssbatchnorm1d import SSBatchNorm1d
from .ssbatchnorm2d import SSBatchNorm2d
from .ssembedding import SSEmbedding
from .utils import Layer_Range

SUPPORT_LAYER = Union[
    SSLinear,
    SSConv2d,
    SSBatchNorm1d,
    SSBatchNorm2d,
    SSEmbedding,
]
