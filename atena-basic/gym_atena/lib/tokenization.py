from collections import defaultdict
import re
from random import shuffle
from bisect import bisect_left, bisect_right
import math

import numpy as np
from scipy import stats

import Utilities.Configuration.config as cfg

import gym_atena.global_env_prop as gep

nan = float('nan')


def tokenize_line_of_column(line, column):
    tokens = set()

    # add token for the whole line
    #if column != "info_line" or "=" not in line:
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


def add_tokens_to_dict(tokens, tokens_dict, len_df):
    for token in tokens:
        # hack for NaN values - see https://stackoverflow.com/questions/45300367/why-adding-multiple-nan-in-python-dictionary-giving-multiple-entries
        #if pd.isnull(token):
        if not isinstance(token, str) and math.isnan(token):
            token = nan
        tokens_dict[token] += 1 / len_df


def tokenize_column(df, column):
    """
    Return a list of (token, frequency) pairs sorted in ascending order of frequency.
    Returning also a sorted list of all frequencies.
    :param df:
    :param column:
    :return:
    """
    tokens_dict = defaultdict(float)
    len_df = len(df)

    if len_df == 0 or column in gep.global_env_prop.env_dataset_prop.DONT_FILTER_FIELDS:
        return [], []

    for idx, line in df[column].iteritems():
        tokens = tokenize_line_of_column(line, column)
        add_tokens_to_dict(tokens, tokens_dict, len_df)

    # sort token_dict by value
    sorted_by_freq_token_frequency_pairs = sorted(tokens_dict.items(), key=lambda kv: kv[1])
    frequencies = [val for token, val in sorted_by_freq_token_frequency_pairs]

    # fig = plt.figure()
    # #ax = plt.gca()
    # frequencies_arr = np.array(frequencies)
    # frequencies_arr_low = frequencies_arr[frequencies_arr <= 0.2]
    # frequencies_arr_high = frequencies_arr[frequencies_arr > 0.2]
    # ax0 = fig.add_subplot(223)
    # ax0.hist(frequencies + [0,1], bins=11)
    # ax1 = fig.add_subplot(211)
    # ax1.hist(frequencies + [0,1], bins=[0]+[0.1/ 2**i for i in range(11, 0, -1)]+[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0])
    # box_cox_frequencies, _ = stats.boxcox(frequencies)
    # normalize_box_cox_frequencies = (box_cox_frequencies - box_cox_frequencies.min()) / (box_cox_frequencies.max() - box_cox_frequencies.min())
    # if len(frequencies_arr_low) > 1:
    #     box_cox_frequencies_low, _ = stats.boxcox(frequencies_arr_low)
    #     if box_cox_frequencies_low.max() > box_cox_frequencies_low.min():
    #         normalize_box_cox_frequencies_low = 0.45 * (box_cox_frequencies_low - box_cox_frequencies_low.min()) / (box_cox_frequencies_low.max() - box_cox_frequencies_low.min())
    #     else:
    #         normalize_box_cox_frequencies_low = frequencies_arr_low
    # else:
    #     normalize_box_cox_frequencies_low = frequencies_arr_low
    #
    # if len(frequencies_arr_high) > 1:
    #     box_cox_frequencies_high, _ = stats.boxcox(frequencies_arr_high)
    #     if box_cox_frequencies_high.max() > box_cox_frequencies_high.min():
    #         normalize_box_cox_frequencies_high = 0.45 * (box_cox_frequencies_high - box_cox_frequencies_high.min()) / (box_cox_frequencies_high.max() - box_cox_frequencies_high.min()) + 0.55
    #     else:
    #         normalize_box_cox_frequencies_high = frequencies_arr_high
    # else:
    #     normalize_box_cox_frequencies_high = frequencies_arr_high
    # ax2 = fig.add_subplot(224)
    # if len(frequencies_arr_low) == 0 or len(frequencies_arr_high) == 0:
    #     ax2.hist(frequencies + [0, 1], bins=11)
    # else:
    #     ax2.hist(list(normalize_box_cox_frequencies_low)+list(normalize_box_cox_frequencies_high)+[0, 1], bins=11)
    #     frequencies = list(normalize_box_cox_frequencies_low)+list(normalize_box_cox_frequencies_high)
    # ax0.set_title("before Box-Cox")
    # ax1.set_title("exponential binning")
    # ax2.set_title("after Box-Cox")
    # plt.show()

    return sorted_by_freq_token_frequency_pairs, frequencies


def get_nearest_neighbor_token(tokens_frequency_lst, frequencies, num):
    """
    Assumes tokens_frequency_lst is sorted. Returns closest value to num.
    If more than one member is equally close, returns random nearest neighbor.
  

    Based on https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    :param tokens_frequency_lst: a list of (token, frequency) pairs
    assumption: sorted by ascending order!
    :param frequencies: a list of frequencies in tokens_frequency_lst in the same order (are given to save the list
    computation running time)
    :param num:
    :return:
    """
    if not tokens_frequency_lst:
        return '<UNK>'

    candidate_nns = _get_nns_candidates(tokens_frequency_lst, frequencies, num)
    if cfg.max_nn_tokens > -1:
        candidate_nns = candidate_nns[:cfg.max_nn_tokens]
    shuffle(candidate_nns)

    #result = min(tokens_lst, key=lambda token: abs(tokens_dict[token]-num))
    # returns the key of the first element
    return candidate_nns[0][0]


def _get_nns_candidates(tokens_frequency_lst, frequencies, num):
    """
    Returns a list of all nearest neighbors (with same distance) from the number
    num.
    :param tokens_frequency_lst:  list of (token, frequency) pairs sorted in ascending order of frequency.
    :param frequencies: The same frequencies in ascending order.
    :param num:
    :return:
    """
    # 'Find leftmost value less than or equal to num'
    right_nearest_pos = bisect_left(frequencies, num)
    # 'Find rightmost value less than or equal to num'
    left_nearest_pos = bisect_right(frequencies, num) - 1

    if right_nearest_pos == len(frequencies):
        left_nearest_pos = bisect_left(frequencies, frequencies[-1])

    elif right_nearest_pos == 0:
        left_nearest_pos = 0
        right_nearest_pos = bisect_right(frequencies, frequencies[0])

    elif left_nearest_pos == right_nearest_pos:
        right_nearest_pos += 1

    elif frequencies[left_nearest_pos] - num < num - frequencies[right_nearest_pos]:
        left_nearest_pos, right_nearest_pos = right_nearest_pos, bisect_right(frequencies,
                                                                              frequencies[right_nearest_pos])

    elif frequencies[left_nearest_pos] - num > num - frequencies[right_nearest_pos]:
        right_nearest_pos, left_nearest_pos = left_nearest_pos + 1, bisect_left(frequencies,
                                                                                frequencies[right_nearest_pos - 1])

    elif frequencies[left_nearest_pos] - num == num - frequencies[right_nearest_pos]:
        right_nearest_pos = bisect_right(frequencies, frequencies[right_nearest_pos])
        left_nearest_pos = bisect_left(frequencies, frequencies[left_nearest_pos])

    else:
        raise RuntimeError("You have a logical error")

    # shuffle list to enable choosing of a random nearest neighbor
    candidate_nns = tokens_frequency_lst[left_nearest_pos: right_nearest_pos]
    return candidate_nns
    

def test_get_nns_candiadtes():
    frequencies = [1, 1, 2, 2, 3, 3, 4, 4]
    num = 0.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [1, 1]
    num = 1
    assert _get_nns_candidates(frequencies, frequencies, num) == [1, 1]
    num = 1.2
    assert _get_nns_candidates(frequencies, frequencies, num) == [1, 1]
    num = 1.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [1, 1, 2, 2]
    num = 1.7
    assert _get_nns_candidates(frequencies, frequencies, num) == [2, 2]
    num = 2
    assert _get_nns_candidates(frequencies, frequencies, num) == [2, 2]
    num = 2.4
    assert _get_nns_candidates(frequencies, frequencies, num) == [2, 2]
    num = 2.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [2, 2, 3, 3]
    num = 2.8
    assert _get_nns_candidates(frequencies, frequencies, num) == [3, 3]
    num = 3
    assert _get_nns_candidates(frequencies, frequencies, num) == [3, 3]
    num = 3.2
    assert _get_nns_candidates(frequencies, frequencies, num) == [3, 3]
    num = 3.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [3, 3, 4, 4]
    num = 3.7
    assert _get_nns_candidates(frequencies, frequencies, num) == [4, 4]
    num = 4
    assert _get_nns_candidates(frequencies, frequencies, num) == [4, 4]
    num = 4.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [4, 4]

    frequencies = [1]
    num = 0.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [1]
    num = 1
    assert _get_nns_candidates(frequencies, frequencies, num) == [1]
    num = 1.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [1]
    

if __name__ == '__main__':
    test_get_nns_candiadtes()
