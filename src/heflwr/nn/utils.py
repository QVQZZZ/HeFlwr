from typing import Tuple, List, Union

# For example, (0, 0.5) represents extracting the first 50%.
Interval = Tuple[str, str]

# For example, [(0, 0.2), (0.5, 0.8)] also represents extracting 50%, but at different positions.
Intervals = List[Tuple[str, str]]

Layer_Range = Union[Interval, Intervals]
