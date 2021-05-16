from atena.simulation.actions import (
    GroupAction,
    Column,
    AggregationFunction,
    BackAction,
    FilterAction,
    FilterOperator,
)


reference1 = [
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_dstport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='SMB'),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.NOTEQUAL, filter_term='SOCKS'),
    GroupAction(grouped_column=Column('length'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]


reference2 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='SMB'),
    BackAction(),
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
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='SOCKS'),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='DATA'),
    BackAction(),
]


reference4 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='TCP'),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_srcport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]


reference5 = [
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='DATA'),
    BackAction(),
    FilterAction(filtered_column=Column('tcp_dstport'), filter_operator=FilterOperator.EQUAL, filter_term='8884'),
    BackAction(),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_dstport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
]

reference6 = [
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='SMB'),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='DATA'),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='TCP'),
    GroupAction(grouped_column=Column('tcp_dstport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]


reference7 = [
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('tcp_stream'), filter_operator=FilterOperator.EQUAL, filter_term='0'),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='DSSETUP'),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='SMB'),
    BackAction(),
]

cyber2_references = [reference1, reference2, reference3, reference4, reference5, reference6, reference7]

assert all([len(reference) == 12 for reference in cyber2_references])
