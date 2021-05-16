import os
import sys
import pickle
from collections import namedtuple, defaultdict
from functools import lru_cache

from cached_property import cached_property
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import seaborn as sns


import gym_atena.lib.helpers as ATENAUtils


import Utilities.Configuration.config as cfg
import gym_atena.global_env_prop as gep



training_dir_path = '/home/oribarel/GIT/ATENA/results/20190518T140227.935305'


class TrainingActionsAnalyzer(object):
    #   ____ ___  _   _ ____ _____ ____  _   _  ____ _____ ___  ____
    #  / ___/ _ \| \ | / ___|_   _|  _ \| | | |/ ___|_   _/ _ \|  _ \
    # | |  | | | |  \| \___ \ | | | |_) | | | | |     | || | | | |_) |
    # | |__| |_| | |\  |___) || | |  _ <| |_| | |___  | || |_| |  _ <
    #  \____\___/|_| \_|____/ |_| |_| \_\\___/ \____| |_| \___/|_| \_\

    def __init__(self, training_dir_path=training_dir_path):
        with open(os.path.join(training_dir_path, 'actions_cntr.pickle'), mode='rb') as handle:
            self.actions_cntr = pickle.load(handle)

        # Update configuration based on schema
        config_file_path = os.path.join(training_dir_path, "args.txt")
        with open(config_file_path, "r") as config_file:
            config_line = config_file.readline()
            config_line = config_line.replace("false", "False")
            config_line = config_line.replace("true", "True")
            config_args = eval(config_line)
            if 'schema' in config_args:
                cfg.schema = config_args["schema"]
            else:
                cfg.schema = 'NETWORKING'
        gep.update_global_env_prop_from_cfg()

        self.map_param_softmax_idx_to_discrete_action = gep.global_env_prop.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION
        self.param_softmax_output_layer_size = gep.global_env_prop.get_parametric_softmax_output_layer_size()

        self.back_actions_count = self.get_back_actions()
        self.filter_actions = self.get_filter_actions()
        self.group_actions = self.get_group_actions()

    @cached_property
    def total_num_of_actions(self):
        return sum(self.actions_cntr.values())

    @cached_property
    def actions_by_type(self):
        filter_actions_cntr = {}
        group_actions_cntr = {}
        back_actions_count = 0

        for idx, count in self.actions_cntr.items():
            action_type, parameters = self.map_param_softmax_idx_to_discrete_action[idx]
            action_type_string = ATENAUtils.OPERATOR_TYPE_LOOKUP[action_type]
            if action_type_string == 'back':
                back_actions_count = count
            elif action_type_string == 'filter':
                # change this if adding more filter operators!
                parameters = list(parameters)
                parameters[1] = (parameters[1] + 1) * 3 - 1
                parameters = tuple(parameters)
                filter_actions_cntr[parameters] = count
            elif action_type_string == 'group':
                group_actions_cntr[parameters] = count
        return back_actions_count, filter_actions_cntr, group_actions_cntr

    def get_back_actions(self):
        return self.actions_by_type[0]

    def get_filter_actions(self):
        return self.actions_by_type[1]

    def get_group_actions(self):
        return self.actions_by_type[2]

    def get_cntr_of_action_type(self, action_type):
        if action_type == 'back':
            return self.back_actions_count
        elif action_type == 'filter':
            return self.filter_actions
        elif action_type == 'group':
            return self.group_actions
        else:
            raise ValueError

    @lru_cache(maxsize=3)
    def get_counts_of_action_type(self, action_type):
        action_type_cntr = self.get_cntr_of_action_type(action_type)
        if isinstance(action_type_cntr, int):
            return action_type_cntr
        else:
            return sum(action_type_cntr.values())

    @lru_cache(maxsize=3)
    def get_percentage_of_action_type(self, action_type):
        """

        Args:
            action_type (str): 'back', 'filter' or 'group'

        Returns:

        """

        return self.get_counts_of_action_type(action_type) / self.total_num_of_actions

    @lru_cache(maxsize=2)
    def get_actions_of_type_by_column_cntr(self, action_type):
        """

        Args:
            action_type: 'filter' or 'group'

        Returns:

        """
        cntr = self.group_actions if action_type == 'group' else self.filter_actions
        map_idx_to_column = gep.global_env_prop.env_dataset_prop.GROUP_COLS if action_type == 'group' else gep.global_env_prop.env_dataset_prop.FILTER_COLS

        count_by_column = defaultdict(int)
        for parameters, count in cntr.items():
            column_idx = parameters[0]
            column_name = map_idx_to_column[column_idx]
            count_by_column[column_name] += count
        return count_by_column

    @cached_property
    def filter_actions_by_operator_cntr(self):
        count_by_operator = defaultdict(int)
        for parameters, count in self.filter_actions.items():
            operator_idx = parameters[1]
            operator_name = ATENAUtils.INT_OPERATOR_MAP_ATENA_STR[operator_idx]
            count_by_operator[operator_name] += count
        return count_by_operator

    @cached_property
    def filter_actions_by_column_and_operator_cntr(self):
        count_by_column_and_operator = defaultdict(int)
        for parameters, count in self.filter_actions.items():
            column_idx = parameters[0]
            operator_idx = parameters[1]
            column_name = gep.global_env_prop.env_dataset_prop.FILTER_COLS[column_idx]
            operator_name = ATENAUtils.INT_OPERATOR_MAP_ATENA_STR[operator_idx]
            count_by_column_and_operator[(column_name, operator_name)] += count
        return count_by_column_and_operator

    @lru_cache(maxsize=2)
    def get_actions_of_type_by_column_percentage(self, action_type):
        """

        Args:
            action_type: 'filter' or 'group'

        Returns:

        """
        result = dict()
        actions_by_column_cntr = self.get_actions_of_type_by_column_cntr(action_type)
        for key, val in actions_by_column_cntr.items():
            result[key] = val / self.get_counts_of_action_type(action_type)

        return result

    @cached_property
    def group_actions_by_column_cntr(self):
        return self.get_actions_of_type_by_column_cntr(action_type='group')

    @cached_property
    def filter_actions_by_column_cntr(self):
        return self.get_actions_of_type_by_column_cntr(action_type='filter')

    @cached_property
    def group_actions_by_column_percentage(self):
        return self.get_actions_of_type_by_column_percentage(action_type='group')

    @cached_property
    def filter_actions_by_column_percentage(self):
        return self.get_actions_of_type_by_column_percentage(action_type='filter')

    @staticmethod
    def show_values_on_bars(ax):
        """
        See https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
        Args:
            ax:

        Returns:

        """
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = p.get_height()
            if value * 100 % 100 != 0:
                value = '{:.2f}'.format(p.get_height())
            else:
                value = int(value)
            ax.text(_x, _y, value, size=12, ha="center")

    @staticmethod
    def plot_bar_plot(names, values, title):
        ax = sns.barplot(x=names, y=values)
        TrainingActionsAnalyzer.show_values_on_bars(ax)

        # Title
        plt.title(title, {'fontsize': 14})

        # change ticks font sizes
        plt.tick_params(labelsize=12.5)

        # Show
        plt.show()

    def plot_counts_per_action_type(self):
        action_types = ['back', 'filter', 'group']
        cntr = {action_type: self.get_counts_of_action_type(action_type)
                for action_type in action_types}
        self.plot_bar_plot(names=action_types,
                           values=[cntr[action_type] for action_type in action_types],
                           title=r'Number of actions per Action Type')

    def plot_counts_actions_of_type_per_column(self, action_type):
        """
        Plot number of occurrences of each column for the given action_type
        Args:
            action_type: 'filter' or 'group'

        Returns:

        """
        actions_by_column_cntr = self.get_actions_of_type_by_column_cntr(action_type=action_type)
        self.plot_bar_plot(names=gep.global_env_prop.env_dataset_prop.KEYS,
                           values=[actions_by_column_cntr[key] for key in
                                   gep.global_env_prop.env_dataset_prop.KEYS],
                           title=r'Number of actions of type {} per column'.format(action_type))

    def plot_counts_filter_per_operator(self):
        filter_actions_per_operator_cntr = self.filter_actions_by_operator_cntr
        self.plot_bar_plot(names=list(filter_actions_per_operator_cntr.keys()),
                           values=list(filter_actions_per_operator_cntr.values()),
                           title=r'Number of Filter actions per operator')

    def plot_counts_filter_per_operator_for_column(self, column):
        cntr = {operator: count for (column_name, operator), count
                in self.filter_actions_by_column_and_operator_cntr.items() if column_name == column}
        if not cntr:
            cntr = {'NO_OPERATOR_USED': 0}
        self.plot_bar_plot(names=list(cntr.keys()),
                           values=list(cntr.values()),
                           title=r'Number of Filter actions on Column {} per operator'.format(column))


if __name__ == '__main__':
    action_types = {'back', 'filter', 'group'}
    analyzer = TrainingActionsAnalyzer(training_dir_path=training_dir_path)

    for action_type in action_types:
        print(action_type)
        print(analyzer.get_percentage_of_action_type(action_type))

    print(analyzer.group_actions_by_column_percentage)
    analyzer.plot_counts_per_action_type()
    analyzer.plot_counts_actions_of_type_per_column(action_type='group')
    analyzer.plot_counts_actions_of_type_per_column(action_type='filter')
    analyzer.plot_counts_filter_per_operator()

    for column in gep.global_env_prop.env_dataset_prop.FILTER_COLS:
        analyzer.plot_counts_filter_per_operator_for_column(column)

