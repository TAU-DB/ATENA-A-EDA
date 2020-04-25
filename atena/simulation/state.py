from collections import namedtuple
from typing import NamedTuple

from atena.simulation.actions import AggregationFunction

FilteringTuple = namedtuple('FilteringTuple', ["column", "term", "operator"])


class AggregationTuple(NamedTuple):
    column: str
    aggregation_function: AggregationFunction


class GetItemByStr(tuple):
    __slots__ = ()

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return self.__getattribute__(attr)
        else:
            return tuple.__getitem__(self, attr)


class AggAttrTuple(namedtuple('AggAttrTuple', ["column", "grouped_or_aggregated_state"]), GetItemByStr):
    """

    e.g.
    ('packet_number', 0.0)
    """
    __slots__ = ()

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)


class GranLayerTuple(namedtuple('GranLayerTuple', ["group_attrs", "agg_attrs",
                                                   "inverse_ngroups", "site_std", "inverse_size_mean"]), GetItemByStr):
    """

    e.g.
    {'group_attrs':
         ['highest_layer', 'tcp_srcport', 'tcp_dstport', 'tcp_stream', 'eth_dst', 'packet_number'],
     'agg_attrs': {'packet_number': 0.0}, 'inverse_ngroups': 1.0, 'site_std': 0.0, 'inverse_size_mean':
         0.16666666666666666}
    """
    __slots__ = ()

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)


class ColumnDataLayerTuple(namedtuple('ColumnDataLayerTuple', ["unique", "nulls", "entropy"]), GetItemByStr):
    """

    e.g.
    {'unique': 1.0, 'nulls': 0.0, 'entropy': 1.0}
    """
    __slots__ = ()

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)


class DisplayTuple(namedtuple('DisplayTuple', ["data_layer", "granularity_layer"]), GetItemByStr):
    """
    e.g.

    {'data_layer': {'packet_number': {'unique': 1.0, 'nulls': 0.0, 'entropy': 1.0}, 'eth_dst':
    {'unique': 0.125, 'nulls': 0.0, 'entropy': 0.0}, 'eth_src': {'unique': 0.125, 'nulls':
    0.0, 'entropy': 0.0}, 'highest_layer': {'unique': 0.25, 'nulls': 0.0, 'entropy':
    0.2704260414863776}, 'info_line': {'unique': 0.75, 'nulls': 0.0, 'entropy': 0.8333333333333334},
    'ip_dst': {'unique': 0.125, 'nulls': 0.0, 'entropy': 0.0}, 'ip_src': {'unique': 0.125,
    'nulls': 0.0, 'entropy': 0.0}, 'length': {'unique': 0.25, 'nulls': 0.0, 'entropy':
    0.2704260414863776}, 'sniff_timestamp': {'unique': 0.5, 'nulls': 0.0, 'entropy':
    0.5833333333333334}, 'tcp_dstport': {'unique': 0.3333333333333333, 'nulls': 0.25,
    'entropy': 0.3868528072345416}, 'tcp_srcport': {'unique': 0.3333333333333333, 'nulls':
    0.25, 'entropy': 0.3868528072345416}, 'tcp_stream': {'unique': 0.3333333333333333,
    'nulls': 0.25, 'entropy': 0.3868528072345416}}, 'granularity_layer': {'group_attrs':
    ['highest_layer', 'tcp_srcport', 'tcp_dstport', 'tcp_stream', 'eth_dst', 'packet_number'],
    'agg_attrs': {'packet_number': 0.0}, 'inverse_ngroups': 1.0, 'site_std': 0.0, 'inverse_size_mean':
    0.16666666666666666}}
    """
    __slots__ = ()

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)


class EnvStateTuple(namedtuple('EnvStateTuple', ["filtering", "grouping", "aggregations"]), GetItemByStr):
    """
    see https://stackoverflow.com/questions/44320382/subclassing-python-namedtuple
    """
    __slots__ = ()

    def reset_filtering(self):
        return self._create_state_tuple(filtering=tuple(),
                                        grouping=self.grouping,
                                        aggregations=self.aggregations)

    def reset_grouping_and_aggregations(self):
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=tuple(),
                                        aggregations=tuple())

    def append_filtering(self, elem):
        field_lst = self._append_to_field(elem, "filtering")
        return self._create_state_tuple(filtering=field_lst,
                                        grouping=self.grouping,
                                        aggregations=self.aggregations)

    def append_grouping(self, elem):
        field_lst = self._append_to_field(elem, "grouping")
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=field_lst,
                                        aggregations=self.aggregations)

    def append_aggregations(self, elem):
        field_lst = self._append_to_field(elem, "aggregations")
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=self.grouping,
                                        aggregations=field_lst)

    def _append_to_field(self, elem, field):
        field_lst = list(self[field])
        field_lst.append(elem)
        return field_lst

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)

    @classmethod
    def _create_state_tuple(cls, filtering, grouping, aggregations):
        return cls(
            filtering=tuple(filtering),
            grouping=tuple(grouping),
            aggregations=tuple(aggregations),
        )

    @classmethod
    def create_empty_state(cls):
        return cls._create_state_tuple((), (), ())
