from enum import Enum
import os
from typing import Union, NewType

import pandas as pd


class CyberDatasetName(Enum):
    DATASET1 = 1
    DATASET2 = 2
    DATASET3 = 3
    DATASET4 = 4


class FlightsDatasetName(Enum):
    DATASET1 = 1
    DATASET2 = 2
    DATASET3 = 3
    DATASET4 = 4


class NetflixDatasetName(Enum):
    DATASET1 = 1


DatasetName = NewType('DatasetName', Union[CyberDatasetName, FlightsDatasetName, NetflixDatasetName])


class SchemaName(Enum):
    CYBER = 'cyber'
    FLIGHTS = 'flights'
    NETFLIX = 'netflix'

    @property
    def dataset_names(self) -> DatasetName:
        if self is SchemaName.CYBER:
            return CyberDatasetName
        elif self is SchemaName.FLIGHTS:
            return FlightsDatasetName
        elif self is SchemaName.NETFLIX:
            return NetflixDatasetName
        else:
            raise NotImplementedError


class DatasetMeta(object):
    def __init__(self, schema: SchemaName, dataset_name: DatasetName):
        self.schema = schema

        # Validate dataset type
        if schema is SchemaName.CYBER:
            assert isinstance(dataset_name, CyberDatasetName)
        elif schema is SchemaName.FLIGHTS:
            assert isinstance(dataset_name, FlightsDatasetName)
        elif schema is SchemaName.NETFLIX:
            assert isinstance(dataset_name, NetflixDatasetName)
        else:
            raise NotImplementedError
        self.dataset_name = dataset_name


class Dataset(object):
    def __init__(self, dataset_meta: DatasetMeta):
        self.dataset_meta = dataset_meta
        self.dataset_df = self.load_data()

    def load_data(self):
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        datasets_path = os.path.join(parent_dir, 'datasets', self.dataset_meta.schema.value)
        dataset_path = os.path.join(datasets_path, f'{self.dataset_meta.dataset_name.value}.tsv')
        df = pd.read_csv(dataset_path, sep='\t', index_col=0)
        return df

    @property
    def columns(self):
        if self.dataset_meta.schema is SchemaName.CYBER:
            return ['packet_number', 'eth_dst', 'eth_src', 'highest_layer', 'info_line',
                    'ip_dst', 'ip_src', 'length',
                    'sniff_timestamp', 'tcp_dstport', 'tcp_srcport',
                    'tcp_stream']
        elif self.dataset_meta.schema is SchemaName.FLIGHTS:
            return ['flight_id', 'airline', 'origin_airport', 'destination_airport', 'flight_number',
                    'delay_reason', 'departure_delay',
                    'scheduled_trip_time',
                    'scheduled_departure', 'scheduled_arrival', 'day_of_week', 'day_of_year']
        elif self.dataset_meta.schema is SchemaName.NETFLIX:
            return ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating',
                    'duration', 'listed_in', 'description']
        else:
            raise NotImplementedError

    @property
    def numeric_columns(self):
        if self.dataset_meta.schema is SchemaName.CYBER:
            return ['packet_number']
        elif self.dataset_meta.schema is SchemaName.FLIGHTS:
            return ['flight_id']
        elif self.dataset_meta.schema is SchemaName.NETFLIX:
            return ['show_id', 'release_year']
        else:
            raise NotImplementedError

    @property
    def primary_key_columns(self):
        if self.dataset_meta.schema is SchemaName.CYBER:
            return ['packet_number', 'length']
        elif self.dataset_meta.schema is SchemaName.FLIGHTS:
            return ['flight_id']
        elif self.dataset_meta.schema is SchemaName.NETFLIX:
            return ['show_id']
        else:
            raise NotImplementedError
