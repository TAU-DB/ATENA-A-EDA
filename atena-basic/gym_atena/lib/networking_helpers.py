import sys
import os
import os.path as path
import math
import operator
from enum import Enum
from functools import lru_cache

from gym_atena.data_schemas.netowrking.columns_data import (KEYS, KEYS_ANALYST_STR, FILTER_COLS, GROUP_COLS,
    NUMERIC_KEYS, AGG_KEYS, AGG_KEYS_ANALYST_STR, FILTER_LIST, FILTER_BY_FIELD_DICT, DONT_FILTER_FIELDS)
from gym_atena.reactida.utils.utilities import Repository
import gym_atena.lib.helpers as ATENAUtils
from gym_atena.envs.env_properties import EnvDatasetProp, BasicEnvProp
from gym_atena.lib.helpers import (
    INT_OPERATOR_MAP_REACT_TO_ATENA,
    INVERSE_AGG_MAP_ATENA,
    OPERATOR_TYPE_LOOKUP,
    INT_OPERATOR_MAP_ATENA,
    AGG_MAP_ATENA,
    normalized_sigmoid_fkt)


@lru_cache()
def create_datasets_repository():
    par_dir = path.dirname(path.dirname(__file__))
    repo_path = path.join(par_dir, 'reactida/session_repositories')
    act_path = path.join(repo_path, 'actions.tsv')
    disp_path = path.join(repo_path, 'displays.tsv')
    datasets_path = path.join(par_dir, 'reactida/raw_datasets_with_pkt_nums')
    # Solving printing bug
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    datasets_repository = Repository(act_path, disp_path, datasets_path)
    sys.stdout = old_stdout
    return datasets_repository


def create_networking_env_properties():
    env_dataset_prop = EnvDatasetProp(
        create_datasets_repository(),
        KEYS,
        KEYS_ANALYST_STR,
        FILTER_COLS,
        GROUP_COLS,
        AGG_KEYS,
        AGG_KEYS_ANALYST_STR,
        NUMERIC_KEYS,
        FILTER_LIST,
        FILTER_BY_FIELD_DICT,
        DONT_FILTER_FIELDS,
    )
    return BasicEnvProp(OPERATOR_TYPE_LOOKUP,
                        INT_OPERATOR_MAP_ATENA,
                        AGG_MAP_ATENA,
                        env_dataset_prop,
                        )


def convert_to_action_vector(action, action_params):
    if action == "back":
        return [0]*6
    if action == "filter":
        col = KEYS.index(action_params["field"])
        condition = INT_OPERATOR_MAP_REACT_TO_ATENA[action_params["condition"]]
        term = FILTER_LIST.index(action_params["term"])
        return [1] + [col] + [condition] + [term] + [0, 0]
    if action == "group":
        col = KEYS.index(action_params["field"])
        if "field" in action_params["aggregations"]:
            agg_col = AGG_KEYS.index(action_params["aggregations"]["field"])
        else:
            agg_col = AGG_KEYS.index('packet_number')  # 'packet_number' col is default aggregation
        if "type" in action_params["aggregations"]:
            agg_func = INVERSE_AGG_MAP_ATENA[(action_params["aggregations"]["type"])]
        else:
            agg_func = INVERSE_AGG_MAP_ATENA['count']  # count is default aggregation
        return [2] + [col] + [0, 0] + [agg_col] + [agg_func]


def compute_normalized_readability_gain(df, prev_prev_df, num_of_grouped_cols):
    """

    Args:
        df:
        prev_prev_df:
        num_of_grouped_cols: 1 if not grouped, else number of grouped columns in df and in prev_prev_df (this is
        the same number, because this method is called after a filter action

    Returns:

    """
    num_of_grouped_cols = 1
    denominator_epsilon = 0.00001
    disp_rows_prev = len(df)
    disp_rows_prev_prev = len(prev_prev_df)

    # how compact is the resulted display
    compact_display_score = normalized_sigmoid_fkt(0.5, 17,
                                                   1 - 1 / math.log(9 + disp_rows_prev * num_of_grouped_cols, 9))
    if disp_rows_prev == 1:
        normalized_readability_gain = -1
    else:
        prev_readability = normalized_sigmoid_fkt(
            0.5, 17, 1 - 1 / math.log(9 + disp_rows_prev * num_of_grouped_cols + denominator_epsilon, 9))
        prev_prev_readability = normalized_sigmoid_fkt(
            0.5, 17, 1 - 1 / math.log(9 + disp_rows_prev_prev * num_of_grouped_cols + denominator_epsilon, 9))
        assert prev_readability >= prev_prev_readability

        # how compact the resulted display of the filter action relative to the display before it.
        readability_gain = 1 - prev_prev_readability / prev_readability

        # transforming the readability gain to be in range [-0.5, 0.5]
        normalized_readability_gain = -0.5 + 1.0 * normalized_sigmoid_fkt(
            0.6, 11, 1 - readability_gain * compact_display_score)
        # making negative normalized_readability_gain in the range [-2.0, 0) instead of
        # of [-0.5, 0) to 'cancel' potential gain of the filter action
        if normalized_readability_gain < 0:
            normalized_readability_gain *= 4
    return normalized_readability_gain
