##########################
#   DISPLAY SIMILARITY   #
##########################


class DisplayDistanceResult(object):
    """
    Helper class to hold the result of a display distance computation
    """

    def __init__(self, data_distance, granularity_distance, display_distance):
        self.data_distance = data_distance
        self.granularity_distance = granularity_distance
        self.display_distance = display_distance


def column_data_difference(col_dict1, col_dict2):
    # measures = ['unique', 'nulls', 'entropy']
    measures = ['unique', 'entropy']

    size = len(measures)
    diff = 0
    for measure in measures:
        v1 = col_dict1[measure]
        v2 = col_dict2[measure]
        if v1 == v2:
            continue
        else:
            diff += 1 / size - min(v1, v2) / (size * max(v1, v2))
    return diff


def data_distance(dl1, dl2, scale=True):
    # Set Difference:
    at1 = set(dl1.keys())
    at2 = set(dl2.keys())
    ikeys = at1.intersection(at2)
    diff_size = len(at1.union(at2) - ikeys)

    raw_dist = diff_size
    for k in ikeys:
        raw_dist += column_data_difference(dl1[k], dl2[k])

    return raw_dist if not scale else (2 * raw_dist) / (len(at1) + len(at2) + raw_dist)


def gran_distance(gl1, gl2, scale=True, same_data=False):
    """

    Args:
        gl1:
        gl2:
        scale:
        same_data (bool): whether the data layer distnce of the two displays is 0 or not.

    Returns:

    """

    group_attr1 = set(gl1['group_attrs'])
    group_attr2 = set(gl2['group_attrs'])
    group_diff = group_attr1.union(group_attr2) - group_attr1.intersection(group_attr2)
    # group_diff_size = len(group_attr1.union(group_attr2) - group_attr1.intersection(group_attr2))
    agg_attr1 = set(gl1['agg_attrs'].keys())
    agg_attr2 = set(gl2['agg_attrs'].keys())
    agg_intersection = agg_attr1.intersection(agg_attr2)
    agg_diff = agg_attr1.union(agg_attr2) - agg_intersection
    group_and_agg_attr1 = group_attr1.union(agg_attr1)
    group_and_agg_attr2 = group_attr2.union(agg_attr2)
    group_and_agg_diff = group_diff.union(agg_diff)
    # remove from agg_intersection all columns that are grouped and aggregated in one display
    # but only aggregated in the other
    agg_intersection = agg_intersection - group_diff
    # agg_diff_size = len(agg_attr1.union(agg_attr2) - agg_intersection)
    # agg_nve_diff = agg_diff_size  # NVE = normalized value entropy
    col_gran_distance = len(group_and_agg_diff)  # NVE = normalized value entropy
    for agg_column in agg_intersection:
        v1 = gl1['agg_attrs'][agg_column]
        v2 = gl2['agg_attrs'][agg_column]
        if v1 == 0 and v2 == 0:
            continue
        elif same_data:
            col_gran_distance += 1 - min(v1, v2) / max(v1, v2)
        else:
            col_gran_distance += 1

    # col_gran_distance = group_diff_size + agg_nve_diff
    # normed_gran_distance = 2 * col_gran_distance / (len(group_attr1)
    #                                                 + len(group_attr2)
    #                                                 +len(agg_attr1)
    #                                                 +len(agg_attr2)
    #                                                 +col_gran_distance)

    normed_gran_distance = 2 * col_gran_distance / (len(group_and_agg_attr1)
                                                    + len(group_and_agg_attr2)
                                                    + col_gran_distance)

    meta_score = 0
    measures = ['inverse_ngroups', 'inverse_size_mean', 'site_std']
    for m in measures:
        if gl1[m] == gl2[m]:
            continue
        else:
            meta_score += 1 / len(measures) - min(gl1[m], gl2[m]) / (len(measures) * max(gl1[m], gl2[m]))

    return ((normed_gran_distance + meta_score) / 2)
    # print(agg_diff_size,group_diff_size)


def display_distance(disp_data1, disp_data2, method="all"):
    # print(disp1.display_id,disp2.display_id)
    data1 = disp_data1["data_layer"]
    data2 = disp_data2["data_layer"]
    data_dist = data_distance(data1, data2)
    if "ds_only" in method:
        return data_dist

    gran1 = disp_data1["granularity_layer"]
    gran2 = disp_data2["granularity_layer"]

    gran1_is_none = (gran1 is None)
    gran2_is_none = (gran2 is None)

    if gran1_is_none and gran2_is_none:  # pd.isnull(gran1) and pd.isnull(gran2):
        granularity_distance = 0.0
    elif gran1_is_none or gran2_is_none:  # pd.isnull(gran1) or pd.isnull(gran2):
        granularity_distance = 1.0
    else:
        granularity_distance = gran_distance(gran1, gran2, same_data=data_dist == 0)

    disp_distance = (3 * data_dist / 2 + granularity_distance) / 2
    return DisplayDistanceResult(data_dist, granularity_distance, disp_distance)


#########################
#   ACTION SIMILARITY   #
#########################


def pair_lca(pair1, pair2):
    k1, v1 = pair1
    k2, v2 = pair2
    if k1 == k2:
        if v1 == v2:
            return (k1, v1)
        else:
            return (k1, None)
    else:
        if v1 == v2:
            return (None, v2)
        else:
            return (None, None)


def set_lca(set1, set2):
    lca = set()
    for pair1 in set1:
        for pair2 in set2:
            p_lca = pair_lca(pair1, pair2)
            if p_lca != (None, None):
                lca.add(p_lca)

    lca_temp = set(lca)

    for pair1 in lca_temp:
        for pair2 in lca_temp:
            if pair1 == pair2:
                continue

            if is_pair_more_general_or_equal(pair1, pair2):
                lca.discard(pair1)

    return lca


def is_pair_more_general_or_equal(pair1, pair2):
    k1, v1 = pair1
    k2, v2 = pair2
    if k1 == k2 and v1 == v2:
        return True
    if k1 is None and v1 is None:
        return True
    if k1 == k2 and v1 is None:
        return True
    if v1 == v2 and k1 is None:
        return True
    return False


def set_dist(set1, set2):
    if len(set2) is not 0:
        K2 = set([x for (x, _) in set2])
        V2 = set([x for (_, x) in set2])
    else:
        K2 = []
        V2 = []
    dist_sum = 0
    for pair1 in set1:
        if pair1 in set2:
            continue
        k1, v1 = pair1
        if k1 in K2:
            if v1 in V2:
                dist_sum += 1
            else:
                dist_sum += 2
        elif v1 in V2:
            dist_sum += 2
        else:
            dist_sum += 3

    return dist_sum


def action_distance(set1, set2, verbose=False):
    s_lca = set_lca(set1, set2)
    if verbose:
        print("LCA:", s_lca)
    d1 = set_dist(set1, s_lca)
    d2 = set_dist(set2, s_lca)
    d3 = set_dist(set1, [])
    d4 = set_dist(set2, [])

    dist = (d1 + d2) / (d3 + d4)
    if verbose:
        print("1 to lca:", d1, "\n2 to lca:", d2, "\n1 to root:", d3, "\n2 to root:", d4)
    return dist

