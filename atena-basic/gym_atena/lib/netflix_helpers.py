import sys
import os
import os.path as path
from functools import lru_cache

import pandas as pd

from gym_atena.data_schemas.netflix.columns_data import (KEYS, KEYS_ANALYST_STR, FILTER_COLS, GROUP_COLS,
    NUMERIC_KEYS, AGG_KEYS, AGG_KEYS_ANALYST_STR, DONT_FILTER_FIELDS)
from gym_atena.envs.env_properties import EnvDatasetProp, BasicEnvProp
from gym_atena.lib.helpers import (
    OPERATOR_TYPE_LOOKUP,
    INT_OPERATOR_MAP_ATENA,
    AGG_MAP_ATENA)


class Repository(object):

    def __init__(self, raw_datasets):
        """

        Args:
            raw_datasets(str): path to datasets
        """
        self.data = []
        self.file_list = os.listdir(raw_datasets)
        self.file_list.sort()
        for f in self.file_list:
            path = os.path.join(raw_datasets, f)
            df = pd.read_csv(path, sep='\t', index_col=0)
            self.data.append(df)


@lru_cache()
def create_datasets_repository():
    par_dir = path.dirname(path.dirname(__file__))
    datasets_path = path.join(par_dir, 'netflix/raw_datasets')
    # Solving printing bug
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    datasets_repository = Repository(datasets_path)
    sys.stdout = old_stdout
    return datasets_repository


def create_netflix_env_properties():
    env_dataset_prop = EnvDatasetProp(
        create_datasets_repository(),
        KEYS,
        KEYS_ANALYST_STR,
        FILTER_COLS,
        GROUP_COLS,
        AGG_KEYS,
        AGG_KEYS_ANALYST_STR,
        NUMERIC_KEYS,
        DONT_FILTER_FIELDS=DONT_FILTER_FIELDS
    )
    return BasicEnvProp(OPERATOR_TYPE_LOOKUP,
                        INT_OPERATOR_MAP_ATENA,
                        AGG_MAP_ATENA,
                        env_dataset_prop,
                        )
