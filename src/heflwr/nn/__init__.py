from .sslinear import SSLinear as SSLinear
from .ssconv2d import SSConv2d as SSConv2d


SUPPORT_LAYER = (
    SSLinear,
    SSConv2d,
)
