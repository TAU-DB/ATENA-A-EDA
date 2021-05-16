"""Evaluating the ability of the agent to act like a human"""

import random
from collections import defaultdict
import pickle
from copy import deepcopy

import numpy as np

from Utilities.Envs.Envs_Utilities import decompress_and_discretize_actions
import gym_atena.lib.helpers as ATENAUtils


def eval_agent_humanity(human_displays_actions_clusters, human_obss, agent_actions, pad_length, env_prop):
    """
    Returns the rate of agent actions that are also performed by humans
    w.r.t. the same observation and also return statistics info

    :param human_displays_actions_clusters:  A dictionary with humans obs-act_lst pairs
    :param human_obss: a list of ndaraay observations to evaluate the agent on
    :param agent_actions: a list of actions of the agent s.t. agent_actions[i] is the action
    of the agent that corresponds to the observation human_obss[i]
    :param pad_length: the length of padding to the agent_actions list
    :return:
    """
    info = {'success_obs': [],
            'failure_obs': [],
            'success_count_per_action_type': defaultdict(int),
            'failure_count_per_action_type': defaultdict(int)
            }

    # remove padded observations and actions
    agent_actions = agent_actions[:len(agent_actions) - pad_length]

    # decompress and discretize agent actions
    agent_discrete_actions = decompress_and_discretize_actions(agent_actions, env_prop)

    # compare agent to humans
    agent_success_count = 0
    for obs, agent_action in zip(human_obss, agent_discrete_actions):
        # check if agent acts like a human on the current observation
        agent_action_type, is_obs_success, obs_success_score = does_agent_acts_on_obs_like_human(agent_action,
                                                                              human_displays_actions_clusters, obs)

        if is_obs_success:  # if agent took action that was taken by human for the current obs
            agent_success_count += 1
            info['success_obs'].append(obs)
            info['success_count_per_action_type'][ATENAUtils.OPERATOR_TYPE_LOOKUP[agent_action_type]] += 1
        else:
            info['failure_obs'].append(obs)
            info['failure_count_per_action_type'][ATENAUtils.OPERATOR_TYPE_LOOKUP[agent_action_type]] += 1

    return agent_success_count / len(human_obss), info


def does_agent_acts_on_obs_like_human(agent_action, human_displays_actions_clusters, obs):
    """
    Returns agent_action_type (a number 0,1,2), is_obs_success (True if agent makes one of human
    actions for the given observation), success_score (a customized success score for acting like
    a human based on the difficulty of the action).
    :param agent_action:
    :param human_displays_actions_clusters:
    :param obs: (ndarray)
    :return:
    """
    is_obs_success = True
    obs = tuple(obs)
    human_actions_for_obs = human_displays_actions_clusters[obs]
    agent_action_type = agent_action[0]
    success_score = 0
    for human_action in human_actions_for_obs:
        human_action_type = human_action[0]
        human_action_col_id = human_action[1]
        agent_action_col_id = agent_action[1]
        human_action_type_str = ATENAUtils.OPERATOR_TYPE_LOOKUP[human_action_type]

        if human_action_type_str == "back":
            if human_action_type == agent_action_type:
                success_score = 1.0
                break

        elif human_action_type_str == "filter":
            if human_action_type == agent_action_type:
                success_score = 0.4
                #human_action_filter_op = ATENAUtils.INT_OPERATOR_MAP_ATENA_STR[human_action[2]]
                #agent_action_filter_op = ATENAUtils.INT_OPERATOR_MAP_ATENA_STR[agent_action[2]]
                if human_action_col_id == agent_action_col_id:
                    #and human_action_filter_op == agent_action_filter_op):
                    success_score += 0.9
                    break

        elif human_action_type_str == "group":
            if human_action_type == agent_action_type:
                success_score = 0.0
                if human_action_col_id == agent_action_col_id:
                    success_score += 1.0
                    break

        else:
            raise ValueError("action should be back, filter or group")
    else:  # if no break
        is_obs_success = False

    return agent_action_type, is_obs_success, success_score


def sample_batch_human_obs(human_obss, batch_size, human_obs_classification_stat=None):
    """

    :param human_obss: (lst of ndarrays)
    :param batch_size: (int)
    :param human_obs_classification_stat: a dictionary that stores for each human observation a tuple
    (obs_success_rate, obs_occurrences+1) s.t. obs_success_rate is the success rate
    of the agent on that obs and obs_occurrences is the number of times the agent
    was presented with this observation. It will be used to generate weights for sampling the
    obs inversely proportional to the success rate
    :return:
    """
    weights = None
    if human_obs_classification_stat:
        # human_obs_classification_stat[tuple(obs)][0] is the success rate
        weights = [1-human_obs_classification_stat[tuple(obs)][0] for obs in human_obss]
    return deepcopy(random.choices(human_obss, k=batch_size, weights=weights))


def get_classification_rewards(human_displays_actions_clusters, batch_obs, batch_act):
    """
    Return a batch of rewards for the classification of the given batch_obs using the agent's
    batch_act
    :param human_displays_actions_clusters:
    :param batch_obs:
    :param batch_act:
    :return:
    """
    batch_size = len(batch_obs)
    rewards = np.zeros(shape=batch_size, dtype=np.float64)

    # decompress and discretize agent actions
    agent_discrete_actions = decompress_and_discretize_actions(batch_act)

    # updating successful classifications
    for idx, (obs, agent_act) in enumerate(zip(batch_obs, agent_discrete_actions)):
        agent_action_type, is_obs_success, obs_success_score = does_agent_acts_on_obs_like_human(agent_act,
                                                                              human_displays_actions_clusters, obs)
        rewards[idx] = obs_success_score

    return rewards
