# coding: utf-8

from copy import deepcopy
import pandas as pd
import json
import numpy as np
import ast
import os
import operator
from .distance import action_distance, display_distance
from collections import defaultdict


def get_dict(dict_str):
    if type(dict_str) is not str:
        return dict_str
    else:
        try:
            return ast.literal_eval(dict_str)
        except:
            print(dict_str)
            return {}


def hack_min(pd_series):
    return np.min(pd_series.dropna())


def hack_max(pd_series):
    return np.max(pd_series.dropna())


INT_OPERATOR_MAP = {
    8: operator.eq,
    32: operator.gt,
    64: operator.ge,
    128: operator.lt,
    256: operator.le,
    512: operator.ne,
}

AGG_MAP = {
    'sum': np.sum,
    'count': len ,
    'min': hack_min,#lambda x:np.nanmin(x.dropna()),
    'max': hack_max,#lambda x:np.nanmax(x.dropna()),
    'avg': np.mean
}

KEYS=[ 'eth_dst', 'eth_src', 'highest_layer', 'info_line',
       'ip_dst', 'ip_src', 'length', 'number',
        'sniff_timestamp', 'tcp_dstport', 'tcp_srcport',
       'tcp_stream']


class Repository:

    def __init__(self, actions_tsv, display_tsv,raw_datasets, schema=KEYS):
        self.actions = pd.read_csv(actions_tsv, sep = '\t', escapechar='\\', keep_default_na=False)
        self.displays = pd.read_csv(display_tsv, sep = '\t', escapechar='\\', keep_default_na=False)
        self.actions.action_params= self.actions.action_params.apply(get_dict)
        #self.actions.bag= self.actions.bag.apply(get_dict)
        self.displays.granularity_layer= self.displays.granularity_layer.apply(get_dict)
        self.displays.data_layer= self.displays.data_layer.apply(get_dict)
        self.schema=schema
        self.data = []
        self.file_list = os.listdir(raw_datasets)
        self.file_list.sort()
        for f in self.file_list:
            path = os.path.join(raw_datasets,f)
            df = pd.read_csv(path, sep = '\t', index_col=0)
            self.data.append(df)


    def get_display_by_id(self, display_id):
        return self.displays[self.displays.display_id == display_id].iloc[0]

    def get_action_by_id(self, action_id):
        return self.actions[self.actions.action_id == action_id].iloc[0]

    def get_session_actions_by_id(self, session_id):
        '''
        Returns a DataFrame containing the "action_type", "action_params", "parent_display_id", "child_display_id"
        coloumns correspoding to this session_id
        '''
        return self.actions[self.actions['session_id'] == session_id][
            ["action_type", "action_params", "parent_display_id", "child_display_id"]]

    @staticmethod
    def static_get_session_actions_by_id_all_rows(session_id, actions_df):
        '''
        Returns a DataFrame containing all action
        rows correspoding to this session_id
        '''
        return actions_df[actions_df['session_id'] == session_id]

    def get_session_actions_by_id_all_rows(self, session_id):
        '''
        Returns a DataFrame containing all action
        rows correspoding to this session_id
        '''
        return self.actions[self.actions['session_id'] == session_id]

    @staticmethod
    def get_dataset_number(session_df):
        return session_df.reset_index(drop=True).at[0, "project_id"]

    @staticmethod
    def get_all_states_of_session(session_df):
        all_states = [{"filtering": [], "grouping": [], "aggregations": []}]
        current_state_history = [{"filtering": [], "grouping": [], "aggregations": []}]
        for index, row in session_df.iterrows():
            Repository.update_history_after_action_and_get_new_state(all_states, current_state_history, row)
        return all_states


    @staticmethod
    def get_all_actions_of_session(session_df):
        return session_df["action_type"].tolist()

    @staticmethod
    def get_all_actions_params_of_session(session_df):
        return session_df["action_params"].tolist()

    @staticmethod
    def update_history_after_action_and_get_new_state(all_states, current_state_history, action):
        '''

        :param all_states: A list of states where each index contains the state corresponding to the index step in the session
        :param current_state_history: A list of states that represents the current stack of states
         (i.e. if back operations were taken, some previous states won't appear here)
        :param action: a single row of an actions_df
        :return: The new state and updates in place (!) both all_states and current_state_history
        '''
        if action.action_type == "back":
            if len(current_state_history) > 1:
                current_state_history.pop()
                new_state = deepcopy(current_state_history[-1])
            else:
                new_state = {"filtering": [], "grouping": [], "aggregations": []}


        elif action.action_type == "filter":
            new_state = deepcopy(current_state_history[-1])
            new_state["filtering"].append(action.action_params)
            current_state_history.append(new_state)

        elif action.action_type == "group":
            # add to the grouping and aggregations lists of the prev state:
            new_state = deepcopy(current_state_history[-1])
            grouping_set = set(new_state["grouping"])
            grouping_params = action.action_params
            grouping_set.add(grouping_params["field"])
            new_state["grouping"] = list(grouping_set)
            agg_dict = grouping_params["aggregations"]
            if agg_dict not in new_state["aggregations"]:
                new_state["aggregations"].extend(agg_dict)
                current_state_history.append(new_state)

        else:
            raise Exception("unknown operator type: " + action.action_type)

        all_states.append(deepcopy(new_state))
        return new_state

    def add_back_actions_to_session(self, session_df):
        '''
        session_df: a DataFrame containing the columns "parent_display_id" and "child_display_id"
        of the all tuples of some session with  and  of some seesion (e.g. by calling get_session_actions_by_id_all_rows)
        '''
        session_df = session_df.reset_index(drop=True)
        session_id, user_id, project_id, solution = tuple(
            session_df.loc[0, ["session_id", "user_id", "project_id", "solution"]].tolist())
        parent_displays = session_df["parent_display_id"].tolist()
        child_displays = session_df["child_display_id"].tolist()
        parent_child_displays = list(zip(parent_displays, child_displays))
        back_rows = defaultdict(list)
        displays_order = []
        idx = 0

        # find all back operations
        for parent_id, child_id in parent_child_displays:
            while displays_order and parent_id != displays_order[-1]:
                back_parent = displays_order.pop()
                back_child = displays_order[-1]
                back_row = pd.DataFrame(
                    {"action_type": "back", "parent_display_id": back_parent, "child_display_id": back_child,
                     "session_id": session_id, "user_id": user_id, "project_id": project_id, "action_params": {},
                     "solution": solution}, index=[idx])
                back_rows[idx].append(back_row)
            if parent_id not in displays_order:
                displays_order.append(parent_id)
            if child_id not in displays_order:
                displays_order.append(child_id)
            idx += 1

        # create a new DataFrame with back operations added
        new_df = df = pd.DataFrame(columns=session_df.columns)
        dfs_to_concat = []
        start_idx = 0
        for idx, back_rows_lst in back_rows.items():
            dfs_to_concat.append(session_df.ix[start_idx:idx - 1])
            for back_row in back_rows_lst:
                dfs_to_concat.append(back_row)
            start_idx = idx
        dfs_to_concat.append(session_df.ix[start_idx:])
        new_df = pd.concat(dfs_to_concat).reset_index(drop=True)

        # reorder DataFrame to get the original columns order
        new_df = new_df[session_df.columns]

        return new_df

    def update_sort_and_project_parents_and_childs(self, session_df):
        """
        Update parent_ids and child ids as if sort and project actions are removed

        :return: new df with updated parent_ids and child_ids
        """
        conversion_dict = dict()  # bad-good pairs
        session_df = session_df.reset_index(drop=True)

        # find first idx of an actions other than sort or project
        good_idx = idx = 0
        good_parent = session_df.at[good_idx, 'parent_display_id']
        first = True
        while good_idx < len(session_df):
            if not first:
                good_parent = session_df.at[good_idx, 'child_display_id']
            while idx < len(session_df) and session_df.at[idx, 'action_type'] in ["project", "sort"]:
                conversion_dict[session_df.at[idx, 'child_display_id']] = good_parent
                idx += 1
            good_idx = idx
            idx = good_idx + 1
            first = False

        # map bad to good
        session_df['child_display_id'] = session_df['child_display_id'].apply(
            lambda x: x if x not in conversion_dict else conversion_dict[x])
        session_df['parent_display_id'] = session_df['parent_display_id'].apply(
            lambda x: x if x not in conversion_dict else conversion_dict[x])

        return session_df, conversion_dict

    def convert_session_df_to_atena(self, session_df):
        # update parent_ids and child_ids
        session_df, _ = self.update_sort_and_project_parents_and_childs(session_df)

        # remove sort and project actions
        session_df = session_df[(session_df["action_type"] != "project") & (session_df["action_type"] != "sort")]

        # add back actions
        session_df = self.add_back_actions_to_session(session_df)

        return session_df

    def convert_actions_to_atena(self):
        session_ids = set(self.actions["session_id"].values)
        flawed_session_ids = set(
            [20, 56, 85, 94, 106, 129, 133, 177, 277, 303, 331, 370, 388, 407, 468, 489, 516, 522, 582, 603, 635,
             723] + list(range(672, 720)))
        session_ids = session_ids - flawed_session_ids
        sessions_dfs = []

        # convert each actions session to ATENA
        for session_id in session_ids:
            converted_df = self.convert_session_df_to_atena(self.get_session_actions_by_id_all_rows(session_id))
            sessions_dfs.append(converted_df)

        return pd.concat(sessions_dfs).reset_index(drop=True)

    def convert_actions_and_save_to_tsv(self):
        converted_df = self.convert_actions_to_atena()
        for col in ['action_id', 'session_id', 'user_id', 'project_id', 'parent_display_id',
                    'child_display_id']:
            converted_df[col] = converted_df[col].fillna(0).astype('int64')
        converted_df.to_csv('actions_atena.tsv', sep='\t', index=False)

    def add_index_column_to_raw_datasets(self):
        for i, df in enumerate(self.data):
            df.to_csv("%d.tsv", sep='\t', index=True)


    def __get_filtered_df(self, project_id, filtering_dict):
    #legacy:
        filters = filtering_dict["list"]
        df = self.data[project_id - 1].copy()
        if filters:
            for filt in filters:
                field = filt["field"]
                op_num = filt["condition"]
                value = filt["term"]
                if op_num in INT_OPERATOR_MAP.keys():
                    opr = INT_OPERATOR_MAP.get(op_num)
                    value= float(value) if df[field].dtype!='O' else value
                    df = df[opr(df[field], value)]
                else:
                    if op_num==16:
                        df = df[df[field].str.contains(value,na=False)]
                    if op_num==2:
                        df = df[df[field].str.startswith(value,na=False)]
                    if op_num==4:
                        df = df[df[field].str.endswith(value,na=False)]

        return df

    def __get_groupby_df(self, df, grouping_dict, aggregation_dict):

        groupings = grouping_dict["list"]
        if aggregation_dict:
            aggregations = aggregation_dict["list"]
            #print(aggregations)
        else:
            aggregations = None
        grouping_attrs = [group["field"] for group in groupings]
        if not grouping_attrs:
            return None,None

        df_gb= df.groupby(grouping_attrs)

        agg_dict={'number':len} #all group-by gets the count by default in REACT-UI
        if aggregations: #Custom aggregations: sum,count,avg,min,max
            for agg in aggregations:
                agg_dict[agg['field']] = AGG_MAP.get(agg['type'])


        agg_df = df_gb.agg(agg_dict)
        return df_gb, agg_df

    def get_raw_display2(self, display_id, pd_group=False):
        row=self.get_display_by_id(display_id)
        raw_df = self.__get_filtered_df(row["project_id"], json.loads(row["filtering"]))
        if type(row["aggregations"]) == float:
            if math.isnan(row["aggregations"]):
                df_gb, agg_df = self.__get_groupby_df(raw_df,json.loads(row["grouping"]), None)
            else:
                df_gb, agg_df = self.__get_groupby_df(raw_df,json.loads(row["grouping"]), json.loads(row["aggregations"]))
        else:
            df_gb, agg_df = self.__get_groupby_df(raw_df,json.loads(row["grouping"]), json.loads(row["aggregations"]))
        if pd_group:
            return raw_df, df_gb
        else:
            return raw_df, agg_df

    def get_raw_display(self, display_id, pd_group=False):
        row=self.get_display_by_id(display_id)
        raw_df = self.__get_filtered_df(row["project_id"], json.loads(row["filtering"]))
        df_gb, agg_df = self.__get_groupby_df(raw_df,json.loads(row["grouping"]), json.loads(row["aggregations"]))
        if pd_group:
            return raw_df, df_gb
        else:
            return raw_df, agg_df

    def __create_action_bag(self,action_id, addType=True):
        action_row= self.get_action_by_id(action_id)
        action_type = action_row.action_type
        action_params = action_row.action_params
        #print(action_params)
        if type(action_params) == str:
            action_params=get_dict(action_params)
        action_bag=set()
        action_bag.add(('type',action_type))
        for k,v in action_params.items():
            if k=='groupPriority':
                continue
            elif k=='aggregations':
                for agg in v:
                    for ak,av in agg.items():
                        if ak=='field':
                            action_bag.add(('agg_field',av))
                        elif ak=='type':
                            action_bag.add(('agg_type',av))

            else:
                action_bag.add((k, v))
        return action_bag

    def display_distance(self,id1, id2):
        d1=self.get_display_by_id(id1)
        d2=self.get_display_by_id(id2)

        dd1={"data_layer": d1.data_layer, "granularity_layer": d1.granularity_layer}
        dd2={"data_layer": d2.data_layer, "granularity_layer": d2.granularity_layer}
        #try:
        return display_distance(dd1,dd2)
        #except:
        #    return None

    def action_distance(self, id1, id2):
        a1 = self.__create_action_bag(id1)
        a2 = self.__create_action_bag(id2)
        try:
            return action_distance(a1, a2)
        except:
            return None
