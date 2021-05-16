import re
import math
from typing import List

import pandas as pd

from atena.simulation.actions import Column

nan = float('nan')


def tokenize_line_of_column(line, column):
    tokens = set()

    # add token for the whole line
    if column != "info_line":
        tokens.add(line)

    if column == "info_line":
        # add token every word between two whitespaces
        for token in line.split():
            token = token.strip(",][)(")
            if "=" not in token and not token.isdigit():  # remove tokens with '='
                tokens.add(token)

        # add [...] tokens
        for token in re.findall('\[.*?\]', line):
            tokens.add(token)

    return tokens


def add_tokens_to_set(tokens, tokens_set):
    for token in tokens:
        # hack for NaN values - see https://stackoverflow.com/questions/45300367/why-adding-multiple-nan-in-python-dictionary-giving-multiple-entries
        if not isinstance(token, str) and math.isnan(token):
            token = nan
        tokens_set.add(token)


def tokenize_column(df: pd.DataFrame, column: Column) -> List[str]:
    """
    Return a list of tokens in the given column in the given DataFrame
    """
    tokens_set = set()
    len_df = len(df)

    if len_df == 0:
        return []

    for idx, line in df[column].iteritems():
        line_tokens = tokenize_line_of_column(line, column)
        add_tokens_to_set(line_tokens, tokens_set)

    return list(tokens_set)
