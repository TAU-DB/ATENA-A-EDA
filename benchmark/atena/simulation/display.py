import logging
import math
import operator
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import entropy


from atena.simulation.actions import FilterOperator, AggregationFunction
from atena.simulation.dataset import Dataset
from atena.simulation.state import FilteringTuple, AggregationTuple, DisplayTuple
from atena.utils.data_structures import CounterWithoutNanKeys

logger = logging.getLogger(__name__)


class DisplayCalculator(object):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @property
    def dataset_df(self) -> pd.DataFrame:
        return self.dataset.dataset_df

    def get_filtered_df(self, filter_tpl: FilteringTuple):
        """

        :param filter_tpl: namedtuple of type FilteringTuple
        :return:
        """
        # legacy:
        if not filter_tpl:
            return self.dataset_df

        df = self.dataset_df.copy()

        for filt in filter_tpl:
            column = filt.column
            filter_operator = filt.operator
            filter_term = filt.term
            assert isinstance(filter_operator, FilterOperator)

            if filter_operator in {FilterOperator.EQUAL, FilterOperator.NOTEQUAL}:
                try:
                    if (df[column].dtype == 'f8' or df[column].dtype == 'u4' or df[column].dtype == 'int64') and filter_term == '<UNK>':
                        filter_term = math.nan
                    filter_term = float(filter_term) if str(df[column].dtype) not in ['object', 'category'] and filter_term != '<UNK>' else filter_term

                    opr = None
                    if filter_operator is FilterOperator.EQUAL:
                        opr = operator.eq
                    elif filter_operator is FilterOperator.NOTEQUAL:
                        opr = operator.ne

                    df = df[opr(df[column], filter_term)]

                except:
                    logger.warning(f"Filter on column {column} with operator {filter_operator.name} and filter_term {filter_term} is emtpy")
                    return df.truncate(after=-1)
            elif filter_operator is FilterOperator.CONTAINS:
                try:
                    if df[column].dtype == 'O' or str(df[column].dtype) == 'category':
                        df = df[df[column].str.contains(filter_term, na=False, regex=False)]
                    elif df[column].dtype == 'f8' or df[column].dtype == 'u4' or df[column].dtype == 'int64':
                        df = df[df[column].astype(str).str.contains(str(filter_term), na=False, regex=False)]
                    else:
                        logger.warning(f"Filter on column {column} with operator {filter_operator.name} and filter_term {filter_term} is emtpy")
                        raise NotImplementedError

                except NotImplementedError:
                    return df.truncate(after=-1)
            else:
                logger.warning(f"Filter on column {column} with operator  {filter_operator.name} and filter_term {filter_term} raised NotImplementedError and will be emtpy")
                raise NotImplementedError

        return df

    def get_groupby_df(self, df: pd.DataFrame, groupings: Iterable[str], aggregations: Iterable[AggregationTuple]):
        if not groupings:
            return None, None

        df_gb = df.groupby(list(groupings), observed=True)

        agg_dict = {}  # A mapping from column name to aggregation function: eg. {'src_ip': len}
        for agg in aggregations:
            assert isinstance(agg.aggregation_function, AggregationFunction)
            agg_dict[agg.column] = agg.aggregation_function.func
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
        # number of unique non NaNs values
        u = len(column_na_value_counts.keys())
        # normalizing the number of unique non-NaNs values by the total number of non-NaNs
        u_n = u / cna_size if u != 0 else 0

        if column.name not in self.dataset.numeric_columns:
            h = entropy(list(column_na_value_counts_values))
            h = h / math.log(cna_size) if cna_size > 1 else 0.0
        else:  # if numeric data only in column
            h = entropy(np.histogram(column_na_value_counts.non_nan_iterable, bins=B)[0]) / math.log(
                B) if cna_size > 1 else 0.0

        return {"unique": u_n, "nulls": n / size, "entropy": h}

    def calc_data_layer(self, df: pd.DataFrame):
        if len(df) == 0:
            ret_dict = {}
            for column in self.dataset.columns:
                ret_dict[column] = {"unique": 0.0, "nulls": 1.0, "entropy": 0.0}
            return ret_dict
        else:
            return {column: self.get_data_column_measures(df[column]) for column in self.dataset.columns}

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
            sizes_mean = sizes.mean()
            inverse_size_mean = 1 / sizes_mean
            if sizes_sum > 0:
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

    def get_raw_data(self, state):
        filtered_df = self.get_filtered_df(state["filtering"])
        gdf, agg_df = self.get_groupby_df(filtered_df, state["grouping"], state["aggregations"])
        return filtered_df, gdf, agg_df

    def get_state_dfs(self, state):
        return self.calculate_display(state)[1]

    def calculate_display(self, state):
        fdf, gdf, adf = self.get_raw_data(state)
        data_layer = self.calc_data_layer(fdf)
        gran_layer = self.calc_gran_layer(gdf, adf)

        disp_tpl = DisplayTuple(data_layer=data_layer, granularity_layer=gran_layer)

        result = disp_tpl, (fdf, adf)
        return result