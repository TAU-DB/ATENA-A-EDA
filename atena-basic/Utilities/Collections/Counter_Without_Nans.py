
"""
A Counter data structure that does not contains NaN as its keys
"""

from collections import Counter
import math


class CounterWithoutNanKeys(Counter):
    def __init__(self, iterable):
        self.non_nan_iterable = [elem for elem in iterable if isinstance(elem, str) or not math.isnan(elem)]
        super().__init__(self.non_nan_iterable)
