from abc import ABC, abstractmethod
from enum import Enum
from typing import NewType

Column = NewType('Column', str)


class FilterOperator(Enum):
    EQUAL = 0
    NOTEQUAL = 1
    CONTAINS = 2


class AggregationFunction(Enum):
    COUNT = 0

    @property
    def func(self):
        if self is AggregationFunction.COUNT:
            return len
        else:
            raise NotImplementedError


class ActionType(Enum):
    BACK = 'BACK'
    FILTER = 'FILTER'
    GROUP = 'GROUP'


class AbstractAction(ABC):
    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    @abstractmethod
    def __repr__(self):
        return NotImplementedError


class BackAction(AbstractAction):
    def __init__(self):
        super().__init__(ActionType.BACK)

    def __repr__(self):
        return 'BACK'


class FilterAction(AbstractAction):
    def __init__(self, filtered_column: Column, filter_operator: FilterOperator, filter_term: str):
        super().__init__(ActionType.FILTER)
        self.filtered_column = filtered_column
        self.filter_operator = filter_operator
        self.filter_term = filter_term

    def __repr__(self):
        return f'FILTER {self.filtered_column} {self.filter_operator}  {self.filter_operator}'


class GroupAction(AbstractAction):
    def __init__(self, grouped_column: Column, aggregated_column: Column, aggregation_function: AggregationFunction):
        super().__init__(ActionType.GROUP)
        self.grouped_column = grouped_column
        self.aggregated_column = aggregated_column
        self.aggregation_function = aggregation_function

    def __repr__(self):
        return f'GROUP {self.grouped_column}'
