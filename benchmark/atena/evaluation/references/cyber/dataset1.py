from atena.simulation.actions import (
    GroupAction,
    Column,
    AggregationFunction,
    BackAction,
    FilterAction,
    FilterOperator,
)

reference1 = [
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('info_line'), filter_operator=FilterOperator.CONTAINS,
                 filter_term='Echo (ping) reply'),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.NOTEQUAL, filter_term='ICMP'),
    GroupAction(grouped_column=Column('tcp_srcport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('ip_src'), filter_operator=FilterOperator.NOTEQUAL,
                 filter_term='192.168.1.122'),
]


reference2 = [
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='ICMP'),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('info_line'), filter_operator=FilterOperator.CONTAINS, filter_term='SYN'),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_dstport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('ip_dst'), filter_operator=FilterOperator.EQUAL, filter_term='82.108.163.88'),
]


reference3 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('ip_src'), filter_operator=FilterOperator.NOTEQUAL, filter_term='192.168.1.122'),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_srcport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]


reference4 = [
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='TCP'),
    GroupAction(grouped_column=Column('tcp_dstport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('ip_dst'), filter_operator=FilterOperator.CONTAINS, filter_term='82.108.87.7'),
    GroupAction(grouped_column=Column('tcp_srcport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]


reference5 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('ip_src'), filter_operator=FilterOperator.EQUAL, filter_term='192.168.1.122'),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('ip_dst'), filter_operator=FilterOperator.EQUAL, filter_term='82.108.87.145'),
    GroupAction(grouped_column=Column('tcp_dstport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),    
]


reference6 = [
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('ip_dst'), filter_operator=FilterOperator.CONTAINS, filter_term='192.168.1.122'),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='ICMP'),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='TCP'),
    GroupAction(grouped_column=Column('tcp_srcport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]


reference7 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('info_line'), filter_operator=FilterOperator.CONTAINS, filter_term='Echo (ping) reply'),
    BackAction(),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('eth_src'), filter_operator=FilterOperator.EQUAL, filter_term='00:26:b9:2b:0b:59'),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_srcport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]


cyber1_references = [reference1, reference2, reference3, reference4, reference5, reference6, reference7]

assert all([len(reference) == 12 for reference in cyber1_references])
