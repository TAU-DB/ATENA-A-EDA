import random

import pandas as pd

from atena.simulation.actions import (
    ActionType,
    BackAction,
    FilterOperator,
    Column,
    FilterAction,
    AggregationFunction, GroupAction,
)
from atena.simulation.dataset import Dataset
from atena.simulation.tokenization import tokenize_column


def get_random_filter_term(df: pd.DataFrame, column: Column) -> str:
    return random.choice(tokenize_column(df, column))


def random_action_generator(dataset: Dataset):
    action_type = random.choice(list(ActionType))

    if action_type is ActionType.BACK:
        return BackAction()
    elif action_type is ActionType.FILTER:
        filtered_column = random.choice(dataset.columns)
        filter_operator = random.choice(list(FilterOperator))
        filter_term = get_random_filter_term(dataset.dataset_df, filtered_column)
        return FilterAction(filtered_column, filter_operator, filter_term)
    elif action_type is ActionType.GROUP:
        grouped_column = random.choice(dataset.columns)
        aggregated_column = random.choice(dataset.primary_key_columns)
        aggregation_function = random.choice(list(AggregationFunction))
        return GroupAction(grouped_column, aggregated_column, aggregation_function)
    else:
        raise NotImplementedError

