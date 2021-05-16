"""
This file defines the properties of the environment (actions supported, types of filter operator supported
 etc.) and the properties of the schema / dataset (column names, Snorkel generative model for the schema etc.).
"""

import math
import os
import sys
import unittest
import logging
from collections import deque
from functools import reduce, lru_cache

import numpy as np
from scipy.stats import entropy

import Utilities.Configuration.config as cfg
from Utilities.Collections.Counter_Without_Nans import CounterWithoutNanKeys
from arguments import FilterTermsBinsSizes
from gym_atena.lib.helpers import (
    DisplayTuple
)

logger = logging.getLogger(__name__)


class EnvDatasetProp(object):
    """
    A Class that holds all properties related to a single dataset / schema
    """

    def __init__(self,
                 repo,
                 KEYS,
                 KEYS_ANALYST_STR,
                 FILTER_COLS,
                 GROUP_COLS,
                 AGG_KEYS,
                 AGG_KEYS_ANALYST_STR,
                 NUMERIC_KEYS,
                 FILTER_LIST=None,
                 FILTER_BY_FIELD_DICT=None,
                 DONT_FILTER_FIELDS=None
                 ):
        """

        Args:
            repo(Repository): an object containing all datasets for the given schema
            KEYS(List[str]): list of all columns in the dataset
            KEYS_ANALYST_STR(List[str]): list of all columns in the same length of KEYS
            such that KEYS_ANALYST_STR[i] is the textual representation of KEYS[i]
            AGG_KEYS(List[str]): list of all columns for which aggregation is allowed
            AGG_KEYS_ANALYST_STR(List[str]): list of all columns in the same length of AGG_KEYS
            such that AGG_KEYS_ANALYST_STR[i] is the textual representation of AGG_KEYS[i]
            NUMERIC_KEYS(List[str]): list of all column containing numerical data on which there
            last action made in the given environment for the given schema
            FILTER_LIST(optional[List[str]]): list of filter terms used by humans
            FILTER_BY_FIELD_DICT(optional[Dict[str,Set[str]): a dictionary mapping each column
            to a
            DONT_FILTER_FIELDS(List[str]):
        """
        self.repo = repo

        self.KEYS = KEYS
        self.KEYS_ANALYST_STR = KEYS_ANALYST_STR
        self.FILTER_COLS = FILTER_COLS
        self.GROUP_COLS = GROUP_COLS
        self.AGG_KEYS = AGG_KEYS
        self.AGG_KEYS_ANALYST_STR = AGG_KEYS_ANALYST_STR
        self.NUMERIC_KEYS = NUMERIC_KEYS
        self.FILTER_LIST = FILTER_LIST if FILTER_LIST is not None else []
        self.FILTER_BY_FIELD_DICT = FILTER_BY_FIELD_DICT if FILTER_BY_FIELD_DICT is not None else {}
        self.DONT_FILTER_FIELDS = DONT_FILTER_FIELDS if DONT_FILTER_FIELDS is not None else {}

        self.KEYS_MAP_ANALYST_STR = {col: col_str for col, col_str in zip(self.KEYS, self.KEYS_ANALYST_STR)}
        self.ANALYST_STR_MAP_KEYS = {col_str: col for col, col_str in zip(self.KEYS, self.KEYS_ANALYST_STR)}
        self.FILTER_COLS_MAP = {col: idx for idx, col in enumerate(self.FILTER_COLS)}

        self.COLS_NO = len(self.KEYS)
        self.AGG_COLS_NO = len(self.AGG_KEYS)
        self.FILTER_TERMS_NO = len(self.FILTER_LIST)


class EnvProp(object):
    ACTION_RANGE = 6.0

    def __init__(self,
                 OPERATOR_TYPE_LOOKUP,
                 INT_OPERATOR_MAP_ATENA,
                 AGG_MAP_ATENA,
                 env_dataset_prop,
                 ):
        """

        Args:
            OPERATOR_TYPE_LOOKUP(Dict[int, str]): a dictionary that maps number to action type
            INT_OPERATOR_MAP_ATENA(Dict[int, Any]): a dictionary that maps number to filter operator
            AGG_MAP_ATENA(Dict[int, Callable]): a dictionary that maps number to aggregation function
            env_dataset_prop(EnvDatasetProp):
        """

        self.OPERATOR_TYPE_LOOKUP = OPERATOR_TYPE_LOOKUP
        self.INT_OPERATOR_MAP_ATENA = INT_OPERATOR_MAP_ATENA
        self.AGG_MAP_ATENA = AGG_MAP_ATENA
        self.env_dataset_prop = env_dataset_prop

        self.ACTION_TYPES_NO = len(self.OPERATOR_TYPE_LOOKUP)
        self.FILTER_OPS = len(self.INT_OPERATOR_MAP_ATENA)
        self.AGG_FUNCS_NO = len(self.AGG_MAP_ATENA)

        # Set filter terms bins
        # MAX_FILTER_TERMS_BY_FIELD_NO = max([len(val) for val in FILTER_BY_FIELD_DICT.values()])
        self.MAX_FILTER_TERMS_BY_FIELD_NO = 1.0  # Maximum continuous value for filter terms (maximum frequency)
        self.bins = self.DISCRETE_FILTER_TERM_BINS_NUM = None
        self.create_filter_term_bins()

        # Setting parametric softmax segments
        self.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION, self.MAP_PARAMETRIC_SOFMAX_DISCRETE_ACTION_TO_IDX = (
            self.get_parametric_softmax_idx_action_mapping())

    def create_filter_term_bins(self):
        """
        A function that change the way in which filter term bins are divided.

        Returns:

        """
        filter_terms_bin_sizes = FilterTermsBinsSizes(cfg.bins_sizes)
        if filter_terms_bin_sizes is FilterTermsBinsSizes.EQUAL_WIDTH:
            self.DISCRETE_FILTER_TERM_BINS_NUM = 11
        elif filter_terms_bin_sizes is FilterTermsBinsSizes.CUSTOM_WIDTH:
            self.bins = [0] + [0.1 / 2 ** i for i in range(11, 0, -1)] + [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                                                          0.5, 0.55,
                                                                          0.6, 0.7, 0.8, 0.9, 1.0]
            self.DISCRETE_FILTER_TERM_BINS_NUM = len(self.bins) - 1
        elif filter_terms_bin_sizes is FilterTermsBinsSizes.EXPONENTIAL:
            self.bins = [None] * cfg.exponential_sizes_num_of_bins
            self.DISCRETE_FILTER_TERM_BINS_NUM = cfg.exponential_sizes_num_of_bins
        else:
            raise NotImplementedError

    def compressed2full_range(self, action_vec, continuous_filter_term=True):
        """
        Change a compressed range vector to full range baed on the range of each entry
        and clip the vector to be in the legal ranges
        :param continuous_filter_term: Boolean whether continuous_filter_term or discrete
        :param action_vec:
        :return:
        """

        RANGE = self.ACTION_RANGE
        entries_ranges = np.array([self.ACTION_TYPES_NO,
                                   self.env_dataset_prop.COLS_NO,
                                   self.FILTER_OPS,
                                   self.MAX_FILTER_TERMS_BY_FIELD_NO,
                                   self.env_dataset_prop.AGG_COLS_NO,
                                   self.AGG_FUNCS_NO])
        full_range = np.multiply(np.array((action_vec + RANGE / 2) / RANGE), entries_ranges) - 0.5
        full_range_filter_term = full_range[3]
        clipped = np.clip(full_range, np.zeros(6), entries_ranges - 1)
        if continuous_filter_term:
            clipped[3] = full_range_filter_term
        return clipped

    def get_parametric_segments(self):
        raise NotImplementedError

    def get_pre_output_layer_size(self):
        """
        Get the size of the pre-output vector for the PARAM_SOFTMAX architecture
        Returns:

        """
        filter_terms_num = self.DISCRETE_FILTER_TERM_BINS_NUM

        return (len(self.OPERATOR_TYPE_LOOKUP) +
                len(self.env_dataset_prop.GROUP_COLS) +
                len(self.env_dataset_prop.FILTER_COLS) +
                len(self.INT_OPERATOR_MAP_ATENA) // 3 +
                filter_terms_num)

    @staticmethod
    def get_seg_size(seg):
        if not seg:
            return 1
        else:
            return reduce(lambda x, y: x * y, seg, 1)

    def get_parametric_softmax_segments_sizes(self):
        parametric_segments = self.get_parametric_segments()
        return [self.get_seg_size(seg) for seg in parametric_segments]

    def get_parametric_softmax_output_layer_size(self):
        return sum(self.get_parametric_softmax_segments_sizes())

    def parametric_softmax_idx_to_discrete_action(self, idx, segs):
        raise NotImplementedError

    def get_softmax_layer_size(self):
        raise NotImplementedError

    def get_parametric_softmax_idx_action_mapping(self):
        parametric_segments = self.get_parametric_segments()
        output_layer_size = self.get_parametric_softmax_output_layer_size()
        idx_to_action = {idx: self.parametric_softmax_idx_to_discrete_action(idx, parametric_segments) for
                         idx in range(output_layer_size)}
        action_to_idx = {act: idx for idx, act in idx_to_action.items()}
        return idx_to_action, action_to_idx

    def static_param_softmax_idx_to_action_type(self, idx):
        """
        Maps an index that represents one off all possible discrete actions in the environment
        to an action type in the environment (i.e. a vector of length 1 that can itself be mapped to an action type)
        Args:
            idx (int): index of an entry in the output vector of an architecture
            of type PARAM_SOFTMAX

        Returns:

        """
        result = np.zeros(1, dtype=np.float32)
        (action_type, parameters) = self.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION[idx]

        result[0] = action_type

        return result

    def get_filtered_df(self, dataset_df, filter_tpl):
        """

        :param dataset_df:
        :param filter_tpl: namedtuple of type FilteringTuple
        :return:
        """
        # legacy:
        if not filter_tpl:
            return dataset_df

        df = dataset_df.copy()

        for filt in filter_tpl:
            field = filt.field
            op_num = filt.condition
            value = filt.term

            opr = self.INT_OPERATOR_MAP_ATENA.get(op_num)
            if opr is not None:
                try:
                    if (df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64') and value == '<UNK>':
                        value = math.nan
                    value = float(value) if str(df[field].dtype) not in ['object', 'category'] and value != '<UNK>' else value

                    '''if opr == operator.ne and pd.isnull(value):
                        df = df[pd.notnull(df[field])]
                    elif opr == operator.eq and pd.isnull(value):
                        df = df[pd.isnull(df[field])]
                    else:'''
                    df = df[opr(df[field], value)]

                except:
                    logger.warning(f"Filter on column {field} with operator {opr} and value {value} is emtpy")
                    return df.truncate(after=-1)
            else:
                # print("***"+field+"\n\n" + str(op_num) +"\n\n" + str(opr)+"\n\n***")
                """
                if op_num==16:
                    df = df[df[field].str.contains(value,na=False)]
                if op_num==2:
                    df = df[df[field].str.startswith(value,na=False)]
                if op_num==4:
                    df = df[df[field].str.endswith(value,na=False)]
                """
                try:
                    # if op_num==6:
                    if op_num in [6, 7, 8]:
                        if df[field].dtype == 'O' or str(df[field].dtype) == 'category':
                            df = df[df[field].str.contains(value, na=False, regex=False)]
                        elif df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64':
                            df = df[df[field].astype(str).str.contains(str(value), na=False, regex=False)]
                        else:
                            logger.warning(f"Filter on column {field} with operator Contains and value {value} is emtpy")
                            raise NotImplementedError
                    elif op_num == 7:
                        df = df[df[field].str.startswith(value, na=False)]
                    elif op_num == 8:
                        df = df[df[field].str.endswith(value, na=False)]
                    else:
                        logger.warning(f"Filter on column {field} with operator number {op_num} and value {value} raised NotImplementedError and will be emtpy")
                        raise NotImplementedError
                except NotImplementedError:
                    return df.truncate(after=-1)

        return df

    def get_groupby_df(self, df, groupings, aggregations):

        if not groupings:
            return None, None

        df_gb = df.groupby(list(groupings), observed=True)

        # agg_dict={'number':len} #all group-by gets the count by default in REACT-UI
        # if aggregations: #Custom aggregations: sum,count,avg,min,max
        agg_dict = {}
        # agg is a namedtuple of type AggregationTuple
        for agg in aggregations:
            agg_dict[agg.field] = self.AGG_MAP_ATENA.get(agg.type)

        try:
            agg_df = df_gb.agg(agg_dict)
        except:
            return None, None
        return df_gb, agg_df

    def get_data_column_measures(self, column):
        """
        for each column, compute its: (1) normalized value entropy (2)Null count (3)Unique values count
        """
        B = 20
        size = len(column)
        if size == 0:
            return {"unique": 0.0, "nulls": 1.0, "entropy": 0.0}
        column_na_value_counts = CounterWithoutNanKeys(column)
        column_na_value_counts_values = column_na_value_counts.values()
        cna_size = sum(column_na_value_counts_values)
        # number of NaNs
        n = size - cna_size
        # u = column.nunique()
        # number of unique non NaNs values
        u = len(column_na_value_counts.keys())
        # normalizing the number of unique non-NaNs values by the total number of non-NaNs
        u_n = u / cna_size if u != 0 else 0

        if column.name not in self.env_dataset_prop.NUMERIC_KEYS:
            # h=entropy(column_na.value_counts(sort=False, dropna=False).values)
            h = entropy(list(column_na_value_counts_values))
            h = h / math.log(cna_size) if cna_size > 1 else 0.0
        else:  # if numeric data only in column
            h = entropy(np.histogram(column_na_value_counts.non_nan_iterable, bins=B)[0]) / math.log(
                B) if cna_size > 1 else 0.0

        return {"unique": u_n, "nulls": n / size, "entropy": h}

    def calc_data_layer(self, df):
        if len(df) == 0:
            ret_dict = {}
            for k in self.env_dataset_prop.KEYS:
                ret_dict[k] = {"unique": 0.0, "nulls": 1.0, "entropy": 0.0}
            return ret_dict
        else:
            # result = df[self.env_dataset_prop.KEYS].apply(self.get_data_column_measures).to_dict()
            return {key: self.get_data_column_measures(df[key]) for key in self.env_dataset_prop.KEYS}

    def get_grouping_measures(self, group_obj, agg_df):
        if group_obj is None or agg_df is None:
            return None

        B = 20
        groups_num = group_obj.ngroups
        if groups_num == 0:
            site_std = 0.0
            inverse_size_mean = 0
            inverse_ngroups = 0
        else:
            sizes = group_obj.size()
            sizes_sum = sizes.sum()
            nsizes = sizes / sizes_sum
            site_std = nsizes.std(ddof=0)
            # size_mean = nsizes.mean()
            sizes_mean = sizes.mean()
            inverse_size_mean = 1 / sizes_mean
            # ngroups=min(len(sizes)/sizes_sum,1)
            if sizes_sum > 0:
                # ngroups = groups_num/sizes_sum
                # Use inverse of ngroups
                inverse_ngroups = 1 / groups_num
            else:
                inverse_ngroups = 0
                inverse_size_mean = 0

        group_keys = group_obj.keys
        agg_keys = list(agg_df.keys())
        agg_nve_dict = {}
        if agg_keys is not None:
            for ak in agg_keys:
                column = agg_df[ak]
                column_na = column.dropna()
                cna_size = len(column_na)
                if cna_size <= 1:
                    agg_nve_dict[ak] = 0.0
                # elif column.name not in NUMERIC_KEYS:
                elif agg_df[ak].dtype == 'O' or str(agg_df[ak].dtype) == 'category':
                    h = entropy(column_na.value_counts().values)
                    agg_nve_dict[ak] = h / math.log(cna_size)
                else:
                    agg_nve_dict[ak] = entropy(np.histogram(column_na, bins=B)[0]) / math.log(B)
        return {"group_attrs": group_keys, "agg_attrs": agg_nve_dict, "inverse_ngroups": inverse_ngroups,
                "site_std": site_std, "inverse_size_mean": inverse_size_mean}

    def calc_gran_layer(self, group_obj, agg_df):
        # print(disp_row.display_id)
        return self.get_grouping_measures(group_obj, agg_df)

    def get_raw_data(self, dataset_df, state):

        filtered_df = self.get_filtered_df(dataset_df, state["filtering"])
        gdf, agg_df = self.get_groupby_df(filtered_df, state["grouping"], state["aggregations"])
        return filtered_df, gdf, agg_df

    def get_state_dfs(self, dataset_df, state, memo=None, dataset_number=None):
        return self.calc_display_vector(dataset_df, state, memo, dataset_number)[2]

    def calc_display_vector(self, dataset_df, state, memo=None, dataset_number=None, step_number=None,
                            states_hist=None, obs_hist=None, len_single_display_vec=None):
        if memo is not None:
            result = memo.get((dataset_number, state))
            if result is not None:  # if in cache
                obs, disp_tpl, dfs = result

                # Stack previous displays to observation
                if states_hist is not None:
                    if len(obs_hist) > 0 and cfg.stack_obs_num > 1:
                        obs_to_stack = obs_hist[-1][-len_single_display_vec * (cfg.stack_obs_num - 1):]
                    else:
                        obs_to_stack = np.empty(shape=0, dtype=np.float32)
                    obs = np.concatenate([obs_to_stack, obs])
                    if len(obs) < len_single_display_vec * cfg.stack_obs_num:
                        obs = np.concatenate(
                            [np.zeros(len_single_display_vec * (
                                    cfg.stack_obs_num - (len(obs) // len_single_display_vec)),
                                      dtype=np.float32), obs]
                        )

                # Add step number to observation
                if cfg.obs_with_step_num and step_number is not None:
                    step_vector = np.zeros(cfg.MAX_NUM_OF_STEPS, np.float32)
                    if step_number != 0:
                        step_vector[step_number - 1] = 1
                    obs = np.concatenate([step_vector, obs])
                result = obs, disp_tpl, dfs
                return result

        fdf, gdf, adf = self.get_raw_data(dataset_df, state)
        data_layer = self.calc_data_layer(fdf)
        gran_layer = self.calc_gran_layer(gdf, adf)

        vlist = []
        for d in data_layer.values():
            vlist += [d["unique"], d["nulls"], d["entropy"]]
            # vlist+=list(d.values())

        if gran_layer is None:
            # The first 0 represents the inverse number of groups (i.e. for 0 groups 1/0 is mapped to 0)
            # The second 0 represents the inverse of groups' mean size (i.e. for mean of 0, 1/0 is mapped to 0)
            vlist += [-1 for _ in self.env_dataset_prop.KEYS] + [0, 0, 0]
        else:
            for k in self.env_dataset_prop.KEYS:
                if k in gran_layer['agg_attrs'].keys():
                    vlist.append(gran_layer['agg_attrs'][k])
                elif k in gran_layer['group_attrs']:
                    vlist.append(2)
                else:
                    vlist.append(-1)

            vlist += [gran_layer['inverse_ngroups'], gran_layer['inverse_size_mean'], gran_layer['site_std']]

        state_obs = np.array(vlist, dtype=np.float32)
        disp_tpl = DisplayTuple(data_layer=data_layer, granularity_layer=gran_layer)

        obs = state_obs
        # Stack previous displays to observation
        if states_hist is not None:
            if len(obs_hist) > 0 and cfg.stack_obs_num > 1:
                obs_to_stack = obs_hist[-1][-len_single_display_vec * (cfg.stack_obs_num - 1):]
            else:
                obs_to_stack = np.empty(shape=0, dtype=np.float32)
            obs = np.concatenate([obs_to_stack, obs])
            if len(obs) < len_single_display_vec * cfg.stack_obs_num:
                obs = np.concatenate(
                    [np.zeros(len_single_display_vec * (
                            cfg.stack_obs_num - (len(obs) // len_single_display_vec)),
                              dtype=np.float32), obs]
                )

        # Add step number to observation
        if cfg.obs_with_step_num and step_number is not None:
            step_vector = np.zeros(cfg.MAX_NUM_OF_STEPS, np.float32)
            if step_number != 0:
                step_vector[step_number - 1] = 1
            obs = np.concatenate([step_vector, state_obs])
        result = obs, disp_tpl, (fdf, adf)

        if memo is not None:
            memo[(dataset_number, state)] = state_obs, disp_tpl, (fdf, adf)

        return result


class BasicEnvProp(EnvProp):
    """
    Environment properties with the following action types:
    1. Group-by + Aggregation
    2. Filter
    3. Back
    """
    def __init__(self,
                 OPERATOR_TYPE_LOOKUP,
                 INT_OPERATOR_MAP_ATENA,
                 AGG_MAP_ATENA,
                 env_dataset_prop,
                 ):
        super().__init__(
            OPERATOR_TYPE_LOOKUP,
            INT_OPERATOR_MAP_ATENA,
            AGG_MAP_ATENA,
            env_dataset_prop,
        )

    def get_softmax_layer_size(self):
        back = 1
        # len(INT_OPERATOR_MAP_ATENA)//3 due to the multiple occurrences (3 times each of 'count', 'neq', 'eq')
        filter_terms_num = self.DISCRETE_FILTER_TERM_BINS_NUM

        filter_ = (
                len(self.env_dataset_prop.FILTER_COLS) *
                len(self.INT_OPERATOR_MAP_ATENA) // 3 *
                filter_terms_num
        )
        group = len(self.env_dataset_prop.GROUP_COLS)

        return back + filter_ + group

    def get_parametric_segments(self):
        """
        Returns a tuple that contains tuple for each action type, that contains in its turn the number of
        options to choose from for each parameter
        Returns:

        """
        back = tuple()
        # len(INT_OPERATOR_MAP_ATENA)//3 due to the multiple occurrences (3 times each of 'count', 'neq', 'eq')
        filter_terms_num = self.DISCRETE_FILTER_TERM_BINS_NUM
        filter_ = tuple([len(self.env_dataset_prop.FILTER_COLS), len(self.INT_OPERATOR_MAP_ATENA) // 3, filter_terms_num])
        group = tuple([len(self.env_dataset_prop.GROUP_COLS)])

        return back, filter_, group

    def parametric_softmax_idx_to_discrete_action(self, idx, segs):
        """
        Maps the given index that represents a discrete action to the action type and its parameters (if any)
        Args:
            idx (int): index of an entry in the output vector of an architecture
                of type PARAM_SOFTMAX
            segs (tuple[tuple[int, optional]]): a tuple that contains tuple for each action type. Each
            such tuple contains the segment size of each parameter (if there is no parameters to the action,
            its tuple is empty)

        Returns: a tuple (action_type, tuple_of_action_parameters)

        """
        cur_idx = 0
        action_type_num = 0
        result = deque()

        for action_num, seg in enumerate(segs):
            seg_size = self.get_seg_size(seg)
            if cur_idx + seg_size > idx:  # we found the right segment
                in_seg_idx = idx - cur_idx
                action_type_num = action_num
                for i in range(len(seg) - 1, -1, -1):
                    result.appendleft(in_seg_idx % seg[i])
                    in_seg_idx //= seg[i]
                break
            else:
                cur_idx += seg_size
        else:  # if we didn't find a suitable segment (if no break)
            raise ValueError("invalid index for the given segments")

        return action_type_num, tuple(result)


# class TestParametricSoftmax(unittest.TestCase):
#
#     def test_parametric_softmax_idx_to_discrete_action(self):
#
#         segs = (tuple(), (12, 3, 11), (12,))
#         segs2 = get_parametric_segments()
#         self.assertEqual(segs, segs2)
#         indices = [0, 1, 11, 12, 13, 33, 330, 350, 372, 396, 397, 408]
#
#         for idx in indices:
#             action_type, parameters = parametric_softmax_idx_to_discrete_action(idx, segs)
#             print("idx: {},   action type: {},   parameters: {}".format(idx, action_type, parameters))
#
#         with self.assertRaises(ValueError):
#             action_type, parameters = parametric_softmax_idx_to_discrete_action(409, segs)
#             print("idx: {},   action type: {},   parameters: {}".format(373, action_type, parameters))
#
#
# if __name__ == '__main__':
#     unittest.main()