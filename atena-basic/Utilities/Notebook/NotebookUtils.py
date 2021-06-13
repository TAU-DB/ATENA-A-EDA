from enum import Enum
import hashlib
import json
import os
import sys
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from uuid import uuid4

import gym
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.util import hash_pandas_object

import Utilities.Configuration.config as cfg
from Utilities.Evaluation.evaluation_measures import draw_nx_display_tree
from Utilities.Utility_Functions import initialize_agent_and_env
from arguments import SchemaName, ArchName
from gym_atena.envs.atena_env_cont import ATENAEnvCont
import gym_atena.lib.helpers as ATENAUtils
from gym_atena.global_env_prop import update_global_env_prop_from_cfg
from gym_atena.lib.networking_helpers import convert_to_action_vector
from gym_atena.reactida.utils.utilities import Repository
from train_agent_chainerrl import act_most_probable



from IPython.display import HTML

HumanStep = namedtuple('HumanStep', 'cur_obs action_vector next_obs')
HumanStepInfo = namedtuple('HumanStepInfo', 'cur_state action_vector action_info next_state dataset_number')

# environment name
env_d = 'ATENAcont-v0'


def get_prev_next_buttons_html_ob_for_display_num(disply_num, run_id):
    prev_and_next_buttons = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>

    a {{
      text-decoration: none;
      display: inline-block;
      padding: 8px 16px;
    }}

    a:hover {{
      background-color: #ddd;
      color: black;
    }}

    .previous {{
      background-color: #f1f1f1;
      color: black;
    }}

    .next {{
      background-color: #4CAF50;
      color: white;
    }}

    .round {{
      border-radius: 50%;
    }}


    </style>
    </head>
    <body>

    <a href="#disp{disply_num-1}_{run_id}" class="previous">&laquo; Previous</a>
    <a href="#disp{disply_num+1}_{run_id}" class="next">Next &raquo;</a>

    <!--<a href="#" class="previous round">&#8249;</a>
    <a href="#" class="next round">&#8250;</a>-->

    </body>
    </html> 
    """

    prev_and_next_buttons_obj = HTML(prev_and_next_buttons)
    return prev_and_next_buttons_obj


def get_new_action_html_obj(display_num, run_id):
    new_action = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    </style>
    </head>
    <body>


    <br style="line-height:5;">
    <hr>
    <h2><a name="disp{display_num}_{run_id}">Action No. {display_num}</a></h2>



    </body>
    </html> 
    """

    new_action_html_obj = HTML(new_action)

    return new_action_html_obj


def get_dataset_number_html_obj(dataset_number, run_id):
    dataset_number_html_txt = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    div.xlarge{{
        font-size:x-large;
        line-height: 1.2;
    }}
    div.large{{
        font-size:large;
        line-height: 1.2;
    }}
    u {{ 
      text-decoration: underline;
    }}
    </style>
    </head>
    <body>


    <a name="disp0_{run_id}"></a>
    <div class=xlarge>
    <p><strong>Running Actions For Dataset No. {dataset_number}</strong></p>
    </div>
    <div class=large>
    <p><strong>The following is the initial display of the dataset:</strong></p>
    </div>



    </body>
    </html> 
    """

    dataset_number_html_obj = HTML(dataset_number_html_txt)

    return dataset_number_html_obj


def get_back_action_html_body():
    back_action = f"""
    <div class=large>
    <p><u>Action Type:</u> <strong>BACK</strong></p>
    </div>
    """    

    return back_action


def get_action_html_obj_helper(html_body):
        html_txt = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        div.large{{
            font-size:medium;
            line-height: 1.2;
        }}
        u {{ 
          text-decoration: underline;
        }}
        </style>
        </head>
        <body>


        {html_body}


        </body>
        </html> 
        """

        result = HTML(html_txt)

        return result


def get_group_action_html_body(grouped_attr, agg_func, agg_attr):
    group_action = f"""
    <div class=large>
    <p><u>Action Type:</u> <strong>GROUP</strong></p>
    <ul><li>Attr: <strong>{grouped_attr}</strong>  &emsp; Agg_func: <strong>{agg_func}</strong>  &emsp; Agg_attr: <strong>{agg_attr}</strong></li></ul>
    </div>
    """

    return group_action


def get_filter_action_html_body(filtered_attr, filter_op, filter_term):
    filter_op_tokens = filter_op.split()
    if len(filter_op_tokens) > 1:
        filter_op = f"""<font color="red">{filter_op_tokens[0]}</font> {''.join(filter_op_tokens[1:])}"""

    filter_action = f"""
    <div class=large>
    <p><u>Action Type:</u> <strong>FILTER</strong></p>
    <ul><li>Attr: <strong>{filtered_attr}</strong>  &emsp; OP: <strong>{filter_op}</strong>  &emsp; Term: <strong>{filter_term}</strong></li></ul>
    </div>
    </html> 
    """

    return filter_action


def get_filtering_state_html_obj(fitering_state):
    filtering_lst = get_filtering_lst_from_state(fitering_state)
    if not filtering_lst:
        return None
    html_body = get_filtering_state_html_body(filtering_lst)

    return get_action_html_obj_helper(html_body)


def get_filtering_lst_from_state(fitering_state):
    filtering_lst = []
    for filtering_tuple in fitering_state:
        filtered_attr = update_global_env_prop_from_cfg().env_dataset_prop.KEYS_MAP_ANALYST_STR[filtering_tuple.field]
        filter_op = ATENAUtils.INT_OPERATOR_MAP_ATENA_PRETTY_STR[filtering_tuple.condition]
        filter_term = filtering_tuple.term
        filtering_lst.append((filtered_attr, filter_op, filter_term))

    return filtering_lst


def get_filtering_state_html_body(filtering_lst):

    if not filtering_lst:
        return ""

    for i, (filtered_attr, filter_op, filter_term) in enumerate(filtering_lst):
        filter_op_tokens = filter_op.split()
        if len(filter_op_tokens) > 1:
            filter_op = f"""<font color="red">{filter_op_tokens[0]}</font> {''.join(filter_op_tokens[1:])}"""
        filtering_lst[i] = (filtered_attr, filter_op, filter_term)

    filter_state = f"""
    <div class=large>
    <u><strong>Filtering State:</strong></u>
    <ul>"""

    for (filtered_attr, filter_op, filter_term) in filtering_lst:
        filter_state += f"""
    <li>Attr: <strong>{filtered_attr}</strong>  &emsp; OP: <strong>{filter_op}</strong>  &emsp; Term: <strong>{filter_term}</strong></li>
    """

    filter_state += """
    </ul>
    </div>


    </body>
    """

    return filter_state


def get_sessions_dfs_from_session_ids(repo, session_ids):
    """
    Return a tuple (solutions_df, solutions_ids) that contains respectively a DataFrame of all actions
    and solution ids of all human sessions with the solution attribute set to True.
    :type repo: Repository
    :param repo:
    :param session_ids_with_react:
    :return:
    """
    # Filtering human sessions with session_id in session_ids_with_react
    solutions_df = repo.actions[repo.actions["session_id"].isin(session_ids)]

    return solutions_df


def get_solution_sessions_dfs_and_id(repo):
    """
    Return a tuple (solutions_df, solutions_ids) that contains respectively a DataFrame of all actions
    and solution ids of all human sessions with the solution attribute set to True.
    :type repo: Repository
    :param repo:
    :return:
    """
    # Filtering human sessions with solution == True
    solutions_df = repo.actions[repo.actions["solution"] == True]

    # Get sessions_id of each session for which solution == True
    solutions_ids = set(solutions_df["session_id"].values)

    return solutions_df, solutions_ids


def run_episode(agent=None,
                dataset_number=None,
                env=None,
                compressed=False,
                filter_by_field=True,
                continuous_filter_term=True,
                actions_lst=None,
                most_probable=False,
                verbose=True
                ):
    """
    Runs a a single episode with the given actions_lst.
    If no actions_lst is given, let the agent choose the actions.
    If no agent is given, sample actions randomly.

    :param agent:
    :param dataset_number:
    :type  env: ATENAEnvCont
    :param env:
    :param compressed:
    :param filter_by_field:
    :param continuous_filter_term:
    :param actions_lst: A list of actions that will be made along the episode
    :param most_probable: If true use most probable actions for agent
    :param verbose
    :return:
    """

    if most_probable:
        assert agent
    info_hist = []

    # Create env if needed
    if env is None:
        env = gym.make(env_d)
    env.render()
    env.reset()

    # Set number of steps in episode
    num_of_steps = cfg.MAX_NUM_OF_STEPS
    if actions_lst is not None:
        num_of_steps = len(actions_lst)

    assert isinstance(env, ATENAEnvCont)

    if isinstance(env, ATENAEnvCont):
        s = env.reset(dataset_number=dataset_number)
        env.max_steps = num_of_steps
    # elif isinstance(env, gym.wrappers.Monitor):
    #     s = env.env.env.reset(dataset_number)
    #     env.env.env.max_steps = num_of_steps
    # else:
    #     s = env.env.reset(dataset_number=dataset_number)

    r_sum = 0
    for ep_t in range(num_of_steps):
        if actions_lst is not None:
            a = actions_lst[ep_t]
        elif not agent:
            a = env.action_space.sample()  # randomly choose an action
        else:
            if most_probable:
                a = act_most_probable(agent, s)
            else:
                a = agent.act(s)
        if verbose:
            print(a)
        s_, r, done, info = env.step(a, compressed=compressed, filter_by_field=filter_by_field,
                                     continuous_filter_term=continuous_filter_term)  # make step in environment
        info_hist.append((info, r))

        if verbose:
            display(info["action"])
        # display(info["raw_display"][1] if info["raw_display"][1] is not None else info["raw_display"][0])
        s = s_
        r_sum += r
        if done:
            break
    return info_hist, r_sum


def get_action_html_obj(raw_action, filter_term):
    html_body = get_action_html_body(raw_action, filter_term)

    return get_action_html_obj_helper(html_body)


def get_action_html_body(raw_action, filter_term):
    env_prop = update_global_env_prop_from_cfg()
    act_string = ATENAUtils.OPERATOR_TYPE_LOOKUP[raw_action[0]]
    if act_string == "back":
        html_body = get_back_action_html_body()
    elif act_string == "group":
        grouped_attr = env_prop.env_dataset_prop.KEYS_ANALYST_STR[raw_action[1]]
        agg_func = ATENAUtils.AGG_MAP_ATENA_STR[raw_action[4]]
        agg_attr = env_prop.env_dataset_prop.AGG_KEYS_ANALYST_STR[raw_action[5]]
        html_body = get_group_action_html_body(grouped_attr, agg_func, agg_attr)
    elif act_string == "filter":
        filtered_attr = env_prop.env_dataset_prop.KEYS_ANALYST_STR[raw_action[1]]
        filter_op = ATENAUtils.INT_OPERATOR_MAP_ATENA_PRETTY_STR[raw_action[2]]
        html_body = get_filter_action_html_body(filtered_attr, filter_op, filter_term)

    assert html_body
    return html_body


def run_episode_analyst_view(agent=None,
                dataset_number=None,
                env=None,
                compressed=False,
                filter_by_field=True,
                continuous_filter_term=True,
                actions_lst=None,
                filter_terms_lst=None,
                angular_grid=False,
                verbose=True
                ):
    """
    Runs a a single episode with the given actions_lst.
    If no actions_lst is given, let the agent choose the actions.
    If no agent is given, sample actions randomly.
    Use an HTML based view for ease of use.

    :param agent:
    :param dataset_number:
    :type  env: ATENAEnvCont
    :param env:
    :param compressed:
    :param filter_by_field:
    :param continuous_filter_term:
    :param actions_lst: A list of actions that will be made along the episode
    :param filter_terms_lst
    :param verbose
    :return:
    """

    if filter_terms_lst is not None:
        assert len(filter_terms_lst) == len(actions_lst)
    track_filter_terms_lst = []

    track_actions_lst = []

    run_id = uuid4()
    info_hist = []

    # Create env if needed
    if env is None:
        env = gym.make(env_d)
    env.render()
    env.reset()

    # Set number of steps in episode
    num_of_steps = cfg.MAX_NUM_OF_STEPS
    if actions_lst is not None:
        num_of_steps = len(actions_lst)

    assert isinstance(env, ATENAEnvCont)

    if isinstance(env, ATENAEnvCont):
        s = env.reset(dataset_number=dataset_number)
        env.max_steps = num_of_steps
    # elif isinstance(env, gym.wrappers.Monitor):
    #     s = env.env.env.reset(dataset_number)
    #     env.env.env.max_steps = num_of_steps
    # else:
    #     s = env.env.reset(dataset_number=dataset_number)

    r_sum = 0
    if verbose:
        display(get_dataset_number_html_obj(dataset_number, run_id))
        display(get_prev_next_buttons_html_ob_for_display_num(0, run_id))
        if not angular_grid:
            display(env.data)
        else:
            display(get_html_ui_grid(env.data, tuple()))
    for ep_t in range(num_of_steps):
        if actions_lst is not None:
            a = actions_lst[ep_t]
        elif not agent:
            a = env.action_space.sample()  # randomly choose an action
        else:
            a = agent.act(s)

        s_, r, done, info = env.step(a, compressed=compressed, filter_by_field=filter_by_field,
                                     continuous_filter_term=continuous_filter_term,
                                     filter_term=None if filter_terms_lst is None else filter_terms_lst[ep_t])  # make step in environment

        track_actions_lst.append(info['raw_action'])

        track_filter_terms_lst.append(info['filter_term'])
      
        if verbose:
            # display action
            display(get_new_action_html_obj(ep_t+1, run_id))
            display(get_action_html_obj(info["raw_action"], info["filter_term"]))
            display(get_prev_next_buttons_html_ob_for_display_num(ep_t+1, run_id))
            
            # display filtering state
            fitering_state = env.states_hisotry[-1].filtering
            filtering_state_html_obj = get_filtering_state_html_obj(fitering_state)
            if filtering_state_html_obj:
                display(filtering_state_html_obj)

            # display tree
            draw_nx_display_tree(track_actions_lst, dataset_number=dataset_number, filter_terms_lst=track_filter_terms_lst if filter_terms_lst is None else filter_terms_lst[:ep_t+1])
            
            # display result
            if ATENAUtils.OPERATOR_TYPE_LOOKUP[info["raw_action"][0]] != "back":
                f, g = info["raw_display"]
                if not angular_grid:
                    if g is not None:
                        df_to_display = g
                    else:
                        df_to_display = f
                    display(df_to_display)
                else:
                    display(get_html_ui_grid(f, env.states_hisotry[-1].grouping))
        # display(info["raw_display"][1] if info["raw_display"][1] is not None else info["raw_display"][0])
        s = s_
        r_sum += r
        if done:
            break
    return info_hist, r_sum


def info_hist_to_raw_actions_lst(info_hist):
    """
    A utility function that returns a list of all actions in the given `info_hist` object
    Args:
        info_hist:

    Returns:

    """
    actions_lst = []
    for info, _ in info_hist:
        info = deepcopy(info)
        info["raw_action"][3] -= 0.5
        actions_lst.append(info["raw_action"])

    return actions_lst


def get_most_probable_actions_lst_of_agent_for_dataset(agent, env, datasat_num):
    """
    Returns a sequence of actions (list) that have highest probability by according to the agent's policy.
    Args:
        agent:
        env:
        datasat_num:

    Returns:

    """
    return get_actions_lst_and_total_reward_of_agent_for_dataset(agent, env, datasat_num, most_probable=True)[0]


def get_actions_lst_of_agent_for_dataset(agent, env, datasat_num):
    """
    Returns a list of actions for the given `env` and `dataset_number` by the given `agent`
    Args:
        agent:
        env:
        datasat_num:

    Returns:

    """
    return get_actions_lst_and_total_reward_of_agent_for_dataset(agent, env, datasat_num)[0]


def get_actions_lst_and_total_reward_of_agent_for_dataset(agent, env, datasat_num, most_probable=False):
    """
    Returns a list of actions for the given `env` and `dataset_number` by the given `agent`
    Args:
        agent:
        env:
        datasat_num:

    Returns:

    """
    info_hist, r_sum = run_episode(agent=agent,
                                   env=env.env,
                                   dataset_number=datasat_num,
                                   most_probable=True,
                                   verbose=False
                                   )
    return info_hist_to_raw_actions_lst(info_hist), r_sum


def simulate(info_hist, displays=False, verbose=True):
    """
    Details about the actions (reward etc.)
    dispays=True will also show the displays (not only the actions)
    """
    #if write_to_markdown_file:
    #    assert markdown_file_descript, ("Passing the argument write_to_markdown_file=True requires"
    #                                    " also passing a markdown_file_descript")

    if verbose:
        for info, r in info_hist:
            info["raw_action"][3] -= 0.5
            print(f'{info["raw_action"]},')


    r_sum = 0
    for i, reward in info_hist:

        if verbose:
            print(f'action: {i["action"]} , reward: {reward}')
            print(f'raw action: {i["raw_action"]}')
        else:
            print(i["action"])
        if verbose:
            print(str(i["reward_info"]))
            print()
        if displays:
            f, g = i["raw_display"]
            if g is not None:
                df_to_display = g
            else:
                df_to_display = f
            display(df_to_display)

        r_sum += reward
    if verbose:
        print("Total Reward:", r_sum)


def analyze_reward(info_hist, actions_lst, summary_reward_data, verbose=True):
    """
    Returns a DataFrame containing a detailed description of the reward given for each action
    in info_hist
    :param info_hist:
    :param actions_lst:
    :param summary_reward_data:
    :param verbose:
    :return:
    """

    # Creating a DataFrame that contains the following columns:
    # action: the action type: `back`, `filter`, `group`
    # action_info: a human readable description of the action
    # column for each reward component that contains the reward from this component
    # total_reward: The total reward in the step due to the various reward components
    actions_info = [info["action"] for info, r in info_hist]
    reward_infos_dict = defaultdict(list)
    reward_infos_dict["action"] = actions_lst
    [reward_infos_dict[key].append(val) for info, r in info_hist for key, val in info["reward_info"].items()]
    [reward_infos_dict["total_reward"].append(r) for info, r in info_hist]
    reward_infos_dict["action_info"] = actions_info
    # [print(str(key) + " " + str(len(val))) for key, val in reward_infos_dict.items()]
    reward_df = DataFrame(reward_infos_dict)

    # Calculating the average reward over the different steps including and excluding `back` actions
    rewards_list = [r for info, r in info_hist]
    rewards_list_no_back = [r for info, r in info_hist if info["action"] != "Back"]
    average_reward_per_action = np.mean(rewards_list)

    # Add average rewards statistics to summary
    average_reward_per_non_back_action = np.mean(rewards_list_no_back)
    summary_reward_data['total_reward'].append(sum(rewards_list))
    summary_reward_data['avg_reward_per_action'].append(average_reward_per_action)
    summary_reward_data['avg_reward_per_non_back_action'].append(average_reward_per_non_back_action)
    summary_reward_data['num_of_actions'].append(len(rewards_list))
    if verbose:
        display(reward_df)
    return reward_df


def are_states_similar(state_dataset_number_pair1, state_dataset_number_pair2):
    """
    Return True if the two states represent the same DataFrame and False otherwise
    :param state_dataset_number_pair1:
    :param state_dataset_number_pair2:
    :return:
    """
    dataset_number1 = state_dataset_number_pair1[1]
    dataset_number2 = state_dataset_number_pair2[1]
    state1 = state_dataset_number_pair1[0]
    state2 = state_dataset_number_pair2[0]
    state1['grouping'].sort()
    state2['grouping'].sort()
    if dataset_number1 != dataset_number2 or state1 != state2:
        # if dataset_number1 != dataset_number2 or state1['filtering'] != state2['filtering']:
        # if dataset_number1 != dataset_number2:
        return False
    return True


def get_scroll_html_obj(html_txt):
    #                 <div style="overflow-x:scroll; width:700px;">
    return f"""
      <div class="div2">
        {html_txt}
      </div>

    """


def fdf_to_json(fdf):
    lst = []
    for i in fdf.index:
        lst.append(fdf.loc[i].to_dict())

    return lst


app_id = str(uuid4()).replace('-', '_')
app_defined = False
def get_html_ui_grid(fdf, grouping_state):
    global app_defined
    schema_name = SchemaName(cfg.schema)
    if schema_name is SchemaName.NETWORKING:
        class ColumnName(Enum):
            packet_number = 'packet_number'
            eth_dst = 'eth_dst'
            eth_src = 'eth_src'
            highest_layer = 'highest_layer'
            info_line = 'info_line'
            ip_dst = 'ip_dst'
            ip_src = 'ip_src'
            length = 'length'
            sniff_timestamp = 'sniff_timestamp'
            tcp_dstport = 'tcp_dstport'
            tcp_srcport = 'tcp_srcport'
            tcp_stream = 'tcp_stream'
    elif schema_name is SchemaName.FLIGHTS:
        class ColumnName(Enum):
            flight_id = 'flight_id'
            airline = 'airline'
            origin_airport = 'origin_airport'
            destination_airport = 'destination_airport'
            flight_number = 'flight_number'
            delay_reason = 'delay_reason'
            departure_delay = 'departure_delay'
            scheduled_trip_time = 'scheduled_trip_time'
            scheduled_departure = 'scheduled_departure'
            scheduled_arrival = 'scheduled_arrival'
            day_of_week = 'day_of_week'
            day_of_year = 'day_of_year'
    else:
        raise NotImplementedError


    class UiGridGroupingState(object):
        def __init__(self):
            self.col_state_dict = {column_name: (False, None) for column_name in ColumnName}
            for grouping_priority, grouped_col in enumerate(grouping_state):
                self.col_state_dict[ColumnName(grouped_col)] = (True, grouping_priority)

        def repr_colomn(self, column_name):
            is_grouped, grouped_priority = self.col_state_dict[column_name]
            if not is_grouped:
                return ''
            return f"""grouping: {{ groupPriority: {grouped_priority}}} ,
             sort: {{ priority: {grouped_priority}, direction: 'asc' }}"""

    ui_grid_grouping_state = UiGridGroupingState()

    # get fdf in json format
    data_json = fdf_to_json(fdf)
    action_id = str(uuid4()).replace('-', '_')
    data_json_str = str(data_json).replace('nan', 'NaN')
    data_json_str = f'data_{action_id} = {data_json_str};'


    start_html = f"""

    <!DOCTYPE html>
    <html ng-app="app_{app_id}">
      <head>
        <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.7.0/angular.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.7.0/angular-touch.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.7.0/angular-animate.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.7.0/angular-aria.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/angular-ui-grid/4.8.1/ui-grid.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/angular-ui-grid/4.8.1/ui-grid.core.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/angular-ui-grid/4.8.1/i18n/ui-grid.tree-base.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/angular-ui-grid/4.8.1/i18n/ui-grid.tree-view.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/angular-ui-grid/4.8.1/ui-grid.css" type="text/css">
            <style>
        .grid {{
      width: 1200px;
      height: 400px;
        }}
        
        .myheader {{
        text-transform: uppercase;
        text-align: center;
        white-space:pre-wrap;
        }}
        
        .capheader
        {{
            text-transform: capitalize;
        }}

    </style> 
    
      </head>
      <body>
    
    <div ng-controller="MainCtrl_{action_id}" style="white-space:pre-wrap">
    
        <div id="grid_{action_id}" ui-grid="gridOptions" ui-grid-grouping ui-grid-auto-resize ui-grid-resize-columns class="grid"></div>
    
    </div>    
    """

    if schema_name is SchemaName.NETWORKING:
        columns_defs = f"""
                            {{
                                displayName: 'Packet No.',
                                field: 'packet_number',
                                filter: {{
                                    condition: 16,
                                }},
                                width: '100',
                                //headerCellClass: $scope.highlightFilteredHeader,
                                sortingAlgorithm: $scope.sortInts,
                                //treeAggregationType: uiGridGroupingConstants.aggregation.COUNT,
                                {ui_grid_grouping_state.repr_colomn(ColumnName.packet_number)}
                            }},
    
                            {{
                                displayName: 'Source MAC',
                                field: 'eth_src',
                                width: '180',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //headerCellClass: 'myheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.eth_src)}
                            }},
                            {{
                                displayName: 'Dest MAC',
                                field: 'eth_dst',
                                width: '180',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //headerCellClass: 'myheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.eth_dst)}
                            }},
                            {{
                                displayName: 'Source IP',
                                field: 'ip_src',
                                width: '150',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //headerCellClass: 'myheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.ip_src)}

                            }},
                            {{
                                displayName: 'Dest IP',
                                field: 'ip_dst',
                                width: '150',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //headerCellClass: 'myheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.ip_dst)}

                            }},
                            {{
                                displayName: 'Src Port',
                                field: 'tcp_srcport',
                                width: '120',
                                headerCellClass: $scope.highlightFilteredHeader,
                                sortingAlgorithm: $scope.sortInts,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.tcp_srcport)}

                            }},
                            {{
                                displayName: 'Dest Port',
                                field: 'tcp_dstport',
                                width: '120',
                                headerCellClass: $scope.highlightFilteredHeader,
                                sortingAlgorithm: $scope.sortInts,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.tcp_dstport)}

                            }},
                            {{
                                name: 'Protocol',
                                field: 'highest_layer',
                                width: '135',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.highest_layer)}
                            }},
                            {{
                                displayName: 'Info',
                                field: 'info_line',
                                width: '800',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.info_line)}

                            }},
                            {{
                                displayName: 'TCP Session',
                                field: 'tcp_stream',
                                width: '150',
                                headerCellClass: $scope.highlightFilteredHeader,
                                sortingAlgorithm: $scope.sortInts,
                                filter: {{
                                    condition: 16
                                }},
                                headerCellClass: 'capheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.tcp_stream)}
    
                            }},
                            {{
                                displayName: 'Length',
                                field: 'length',
                                width: '80',
                                headerCellClass: $scope.highlightFilteredHeader,
                                sortingAlgorithm: $scope.sortInts,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.length)}
                            }},
                            {{
                                displayName: 'TimeStamp',
                                field: 'sniff_timestamp',
                                width: '100',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //sort: {{
                                  //  direction: uiGridConstants.DESC
                                //}},
                                sortingAlgorithm: $scope.sortDates,
                                {ui_grid_grouping_state.repr_colomn(ColumnName.sniff_timestamp)}

                            }}
"""
    elif schema_name is SchemaName.FLIGHTS:
        columns_defs = f"""
                            {{
                                displayName: 'Flight ID',
                                field: 'flight_id',
                                filter: {{
                                    condition: 16,
                                }},
                                width: '100',
                                //headerCellClass: $scope.highlightFilteredHeader,
                                sortingAlgorithm: $scope.sortInts,
                                //treeAggregationType: uiGridGroupingConstants.aggregation.COUNT,
                                {ui_grid_grouping_state.repr_colomn(ColumnName.flight_id)}
                            }},
    
                            {{
                                displayName: 'Airline',
                                field: 'airline',
                                width: '180',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //headerCellClass: 'myheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.airline)}
                            }},
                            {{
                                displayName: 'Origin Airport',
                                field: 'origin_airport',
                                width: '180',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //headerCellClass: 'myheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.origin_airport)}
                            }},
                            {{
                                displayName: 'Destination Airport',
                                field: 'destination_airport',
                                width: '150',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //headerCellClass: 'myheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.destination_airport)}

                            }},
                            {{
                                displayName: 'Flight Number',
                                field: 'flight_number',
                                width: '150',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //headerCellClass: 'myheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.flight_number)}

                            }},
                            {{
                                displayName: 'Delay Reason',
                                field: 'delay_reason',
                                width: '175',
                                headerCellClass: $scope.highlightFilteredHeader,
                                //sortingAlgorithm: $scope.sortInts,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.delay_reason)}

                            }},
                            {{
                                displayName: 'Departure Delay',
                                field: 'departure_delay',
                                width: '160',
                                headerCellClass: $scope.highlightFilteredHeader,
                                //sortingAlgorithm: $scope.sortInts,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.departure_delay)}

                            }},
                            {{
                                name: 'Scheduled Trip Time',
                                field: 'scheduled_trip_time',
                                width: '170',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.scheduled_trip_time)}
                            }},
                            {{
                                displayName: 'Scheduled Departure',
                                field: 'scheduled_departure',
                                width: '170',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.scheduled_departure)}

                            }},
                            {{
                                displayName: 'Scheduled Arrival',
                                field: 'scheduled_arrival',
                                width: '150',
                                headerCellClass: $scope.highlightFilteredHeader,
                                //sortingAlgorithm: $scope.sortInts,
                                filter: {{
                                    condition: 16
                                }},
                                headerCellClass: 'capheader',
                                {ui_grid_grouping_state.repr_colomn(ColumnName.scheduled_arrival)}
    
                            }},
                            {{
                                displayName: 'Day of Week',
                                field: 'day_of_week',
                                width: '140',
                                headerCellClass: $scope.highlightFilteredHeader,
                                sortingAlgorithm: $scope.sortInts,
                                filter: {{
                                    condition: 16
                                }},
                                {ui_grid_grouping_state.repr_colomn(ColumnName.day_of_week)}
                            }},
                            {{
                                displayName: 'Day of Year',
                                field: 'day_of_year',
                                width: '140',
                                headerCellClass: $scope.highlightFilteredHeader,
                                filter: {{
                                    condition: 16
                                }},
                                //sort: {{
                                  //  direction: uiGridConstants.DESC
                                //}},
                                //sortingAlgorithm: $scope.sortDates,
                                sortingAlgorithm: $scope.sortInts,
                                {ui_grid_grouping_state.repr_colomn(ColumnName.day_of_year)}

                            }}
        """
    else:
        raise NotImplementedError


    angular_scripts1 = f"""
    <script>
    {'//' if app_defined else ''}var app_{app_id} = angular.module('app_{app_id}', ['ngAnimate', 'ngTouch', 'ui.grid', 'ui.grid.grouping',  'ui.grid.resizeColumns', 'ui.grid.autoResize']);
    
    app_{app_id}.controller('MainCtrl_{action_id}', ['$scope', '$http', '$interval', 'uiGridGroupingConstants', function ($scope, $http, $interval, uiGridGroupingConstants ) {{
    
      $scope.gridOptions = {{
    
    enableHorizontalScrollbar: 2,
                        fastWatch: true,
                        enableSorting: true,
                        //useExternalFiltering: true,
                        enableColumnResizing: true,
                        enableFiltering: true,
                        enableGridMenu: true,
                        enableRowSelection: true,
                        enableRowHeaderSelection: false,
                        multiSelect: true,
                        modifierKeysToMultiSelect: true,
    columnDefs: [
                            {columns_defs}
                            ,
    
                        ],
    
        onRegisterApi: function( gridApi ) {{
          $scope.gridApi = gridApi;
        }}
      }};
    
    """


    angular_scripts2 = f"""
      $scope.gridOptions.data = data_{action_id};
       //$scope.gridApi.grouping.clearGrouping();
        //$scope.gridApi.grouping.groupColumn('highest_layer');
        //$scope.gridApi.grouping.aggregateColumn('packet_number', uiGridGroupingConstants.aggregation.COUNT);
    
       $scope.toggleRow = function( rowNum ){{
        $scope.gridApi.treeBase.toggleRowTreeState($scope.gridApi.grid.renderContainers.body.visibleRowCache[rowNum]);
      }}; 
      $scope.toggle = function(){{
      num={len(grouping_state)};
      var i;
    
      node_list0=$scope.gridApi.grid.renderContainers.body.visibleRowCache;
      for (i = 1; i < num; i++){{
      node_list1=[];
    
      angular.forEach(node_list0,function(row){{
          $scope.gridApi.treeBase.expandRow(row);
          node_list1=node_list1.concat($scope.gridApi.treeBase.getRowChildren(row))
          //alert(node_list1.length);
      }});    
         node_list0=node_list1;
         //alert(node_list1.length);
    
    
      }};
      }};
    
      angular.element(function () {{
        console.log('page loading completed');
        $scope.toggle()
    }})
    
      }}]);
    
    
      </script>
      """



    end_html = """
      </body>
    </html>
    
    """
    app_defined = True
    all_html = start_html + angular_scripts1 + data_json_str + angular_scripts2 + end_html
    return HTML(all_html)


#####################################
# Recommender System
####################################


class RecommenderAgentStepResult(object):
    def __init__(self, info, reward, verbose=True):
        self.info = info
        self.reward = reward
        self.verbose = verbose

    @property
    def reward_info(self):
        return self.info["reward_info"]

    @property
    def df_to_display(self):
        f, g = self.info["raw_display"]
        if g is not None:
            df_to_display = g
        else:
            df_to_display = f
        return df_to_display


class RecommenderAgent(object):
    class Cfg(object):
        def __init__(
                self,
                arch,
                obs_with_step_num,
                stack_obs_num,
        ):
            self.arch = arch
            self.obs_with_step_num = obs_with_step_num
            self.stack_obs_num = stack_obs_num

        @classmethod
        def create_from_cfg(cls, cfg_module):
            return cls(
                cfg_module.arch,
                cfg_module.obs_with_step_num,
                cfg_module.stack_obs_num,
            )

    def __init__(
            self,
            model_dir_path,
            dataset_number,
            env=None,
            filter_by_field=True,
            most_probable=False,
            verbose=True
    ):
        # Create agent and set configuration accordingly
        self.agent = self._create_agent(model_dir_path)

        self.state = None
        # Create env if needed
        if env is None:
            env = self._create_env(dataset_number)
        self.env = env

        self.agent_cfg = self.Cfg.create_from_cfg(cfg)
        self.most_probable = most_probable
        self.filter_by_field = filter_by_field
        self.verbose = verbose

        self.last_recommendation = (None, None)  # (action, filter_term)

    def _create_env(self, dataset_number):
        # Create env if needed
        env = gym.make(env_d)
        env.render()
        env.reset()

        # Set number of steps in episode
        num_of_steps = 1000

        assert isinstance(env, ATENAEnvCont)

        self.state = env.reset(dataset_number=dataset_number)
        env.max_steps = num_of_steps

        return env

    def _create_agent(self, model_dir_path):
        command = f"train.py --env ATENAcont-v0 --demo --load {model_dir_path}"

        sys.argv = command.split()
        agent, env, args = initialize_agent_and_env()

        return agent

    def apply_custom_action(self, action_vec, filter_term=None):
        self.set_human_framework_config()

        # make step in the environment
        self.state, reward, done, info = self.env.step(
            action_vec, compressed=False, filter_by_field=False,
            continuous_filter_term=False, filter_term=filter_term
        )

        return RecommenderAgentStepResult(info, reward, self.verbose)

    def get_agent_action(self):
        if self.most_probable:
            action = act_most_probable(self.agent, self.state)
        else:
            action = self.agent.act(self.state)

        self.last_recommendation = (action, self._get_filter_term(action))
        return action

    def _get_filter_term(self, action):
        action_vec, _ = self.env.action_to_vec(action, filter_by_field=self.filter_by_field)

        filter_term = None
        operator_type = self.env.env_prop.OPERATOR_TYPE_LOOKUP.get(action_vec[0])
        if operator_type == 'filter':
            col = self.env.env_dataset_prop.KEYS[action_vec[1]]
            filter_term = self.env.compute_nearest_neighbor_filter_term(action_vec, col)
        return filter_term

    def get_agent_action_str(self):
        self.set_agent_framework_config()

        action = self.get_agent_action()
        filter_term = self._get_filter_term(action)
        action_vec, _ = self.env.action_to_vec(action, filter_by_field=self.filter_by_field)

        return self.env.translate_action(action_vec, filter_term=filter_term)

    def apply_agent_action(self, use_last_recommendation=True):
        self.set_agent_framework_config()

        action = self.last_recommendation[0] if use_last_recommendation else self.get_agent_action()
        filter_term = self.last_recommendation[1] if use_last_recommendation else None
        self.state, reward, done, info = self.env.step(
            action, compressed=False, filter_by_field=self.filter_by_field,
            continuous_filter_term=True, filter_term=filter_term
        )

        return RecommenderAgentStepResult(info, reward, self.verbose)

    @property
    def original_dataset(self):
        return self.env.data

    def set_agent_framework_config(self):
        cfg.arch = self.agent_cfg.arch
        cfg.obs_with_step_num = self.agent_cfg.obs_with_step_num
        cfg.stack_obs_num = self.agent_cfg.stack_obs_num

    def set_human_framework_config(self):
        cfg.arch = ArchName.FF_GAUSSIAN.value
        cfg.obs_with_step_num = False
        cfg.use_snorkel = False
