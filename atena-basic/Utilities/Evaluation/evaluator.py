# from collections import defaultdict
# from enum import Enum
# import os
# import sys
# from functools import lru_cache
#
# import gym
# # from IPython.core.display import display
#
# import gym_atena.envs.atena_env_cont as atena_env_cont
# import gym_atena.lib.networking_helpers
# import pandas as pd
# import numpy as np
import logging
import math

from Utilities.Utility_Functions import (
    initialize_agent_and_env,
)
from Utilities.Evaluation.evaluation_measures import (
    remove_back_tokens_from_nested_lists,
    corpus_bleu_without_back,
    sentence_bleu_without_back,
    tree_corpus_gleu_n,
    tree_sentence_gleu_n,
    tree_corpus_bleu_n,
    tree_sentence_bleu_n,
    get_refs_and_cands_tokens_from_actions,
    get_refs_and_multiple_cands_tokens_from_actions,
    compute_minimum_display_TED_from_actions,
    draw_ref_and_candidate_trees_side_by_side,
    precision_score_without_back,
    recall_score_without_back,
    f1_score_without_back,
    micro_precision_without_back,
    micro_recall_without_back,
    micro_f1_without_back,
    paired_pvalue
)
import Utilities.Configuration.config as cfg
from arguments import ArchName, SchemaName
from Utilities.Notebook.NotebookUtils import *
import gym_atena.lib.helpers as ATENAUtils
import gym_atena.global_env_prop as gep

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import SmoothingFunction

logger = logging.getLogger(__name__)


class TedResult(object):
    """
    A helper class to hold the result  of tree edit distance calculation
    """
    def __init__(self, min_ted, argmin_ref, teds_lst, min_ted_edit_operations, normalized,
                 is_empty=False):
        self.min_ted = min_ted  # The minimal tree edit distance
        self.argmin_ref = argmin_ref  # The index of reference for which we got the minimal tree edit distance
        self.teds_lst = teds_lst  # A list of tree edit distances for each reference
        self.min_ted_edit_operations = min_ted_edit_operations  # The edit operations for the reference with minimum tree edit distance
        self.normalized = normalized  # Whether or not the tree edit distance is normalized
        self.is_empty = is_empty  # Whether or not the calculation is being done for an empty candidate.

    @classmethod
    def get_empty_result(cls, normalized):
        """
        REturn a dummy result for an empty candidate
        Args:
            normalized:

        Returns:

        """
        return cls(math.nan, None, [math.nan], None, normalized=normalized, is_empty=True)


class AllDatasetsTedResult(object):
    """
    A helper function to store tree edit distance scores for candidate of each dataset.
    """
    def __init__(self):
        """
        In the following list the ith index contains the relevant value for the ith dataset
        """
        # Unnormalized
        self.ted_results_lst = []
        self.min_teds = []
        self.avg_teds = []

        # Normalized
        self.norm_ted_results_lst = []
        self.norm_min_teds = []
        self.norm_avg_teds = []

    def append(self, ted_result):
        if not ted_result.normalized:
            self.ted_results_lst.append(ted_result)
            self.min_teds.append(ted_result.min_ted)
            self.avg_teds.append(np.mean(ted_result.teds_lst))
        elif ted_result.normalized:
            self.norm_ted_results_lst.append(ted_result)
            self.norm_min_teds.append(ted_result.min_ted)
            self.norm_avg_teds.append(np.mean(ted_result.teds_lst))

    def get_eval_metric_lst(self, eval_metric):
        """
        Returns the relevant list for the given tree edit distance EvalMetric
        Args:
            eval_metric:

        Returns:

        """
        if eval_metric is EvalMetric.MIN_TED:
            teds_lst = self.min_teds
        elif eval_metric is EvalMetric.NORM_MIN_TED:
            teds_lst = self.norm_min_teds
        elif eval_metric is EvalMetric.AVG_TED:
            teds_lst = self.avg_teds
        elif eval_metric is EvalMetric.NORM_AVG_TED:
            teds_lst = self.norm_avg_teds
        return teds_lst

    def get_average_all_datasets(self, eval_metric):
        teds_lst = self.get_eval_metric_lst(eval_metric)
        teds_to_avg = []
        for ted, ted_result in zip(teds_lst, self.ted_results_lst):
            if not ted_result.is_empty:
                teds_to_avg.append(ted)
        return np.mean(teds_to_avg)


class EvalMethod(Enum):
    REFERENCE = 'reference'
    GREEDY = 'greedy'
    INTERESTINGNESS_GREEDY = 'interestingness_greedy'  # A greedy agent with interestingness component only
    RANDOM = 'random'
    K_RANDOM = 'k_random'
    MAX_K_RANDOM = 'max_k_random'
    MOST_PROBABLE = 'most_probable'
    INTER_MOST_PROBABLE = 'inter_most_probable'  # The most probable policy of an agent with interestingness component only
    SOFTMAX_MOST_PROBABLE = 'softmax_most_probable'  # The most probable policy of an agent with softmax architecture and binning for filter terms
    SOFTMAX_LIST_MOST_PROBABLE = 'softmax_list_most_probable'  # The most probable policy of an agent with softmax architecture and list of filter terms


class EvalMetric(Enum):
    BLEU1 = 'BLEU1'
    BLEU2 = 'BLEU2'
    BLEU3 = 'BLEU3'
    GLEU1 = 'GLEU1'
    GLEU2 = 'GLEU2'
    GLEU3 = 'GLEU3'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'F1'
    MICRO_PRECISION = 'micro-precision'
    MICRO_RECALL = 'micro-recall'
    MICRO_F1 = 'micro-F1'
    MIN_TED = 'min TED'
    NORM_MIN_TED = 'norm min TED'
    AVG_TED = 'avg TED'
    NORM_AVG_TED = 'norm avg TED'

    @staticmethod
    def map_num_to_bleu_eval_metric(num):
        if num == 1:
            return EvalMetric.BLEU1
        elif num == 2:
            return EvalMetric.BLEU2
        elif num == 3:
            return EvalMetric.BLEU3
        else:
            raise NotImplementedError

    @staticmethod
    def map_bleu_eval_metric_to_num(eval_metric):
        if eval_metric in [EvalMetric.BLEU1, EvalMetric.GLEU1]:
            return 1
        elif eval_metric in [EvalMetric.BLEU2, EvalMetric.GLEU2]:
            return 2
        elif eval_metric in [EvalMetric.BLEU3, EvalMetric.GLEU3]:
            return 3
        else:
            raise NotImplementedError

    @staticmethod
    def map_gleu_eval_metric_to_num(eval_metric):
        if eval_metric is EvalMetric.GLEU1:
            return 1
        elif eval_metric is EvalMetric.GLEU2:
            return 2
        elif eval_metric is EvalMetric.GLEU3:
            return 3
        else:
            raise NotImplementedError

    @staticmethod
    def map_bleu_eval_metric_to_corpus_str(eval_metric):
        if eval_metric in [EvalMetric.BLEU1, EvalMetric.GLEU1]:
            return 'corpus_tree_bleu1'
        elif eval_metric in [EvalMetric.BLEU2, EvalMetric.GLEU2]:
            return 'corpus_tree_bleu2'
        elif eval_metric in [EvalMetric.BLEU3, EvalMetric.GLEU3]:
            return 'corpus_tree_bleu3'
        else:
            raise NotImplementedError

    @classmethod
    def get_bleu_and_f1_eval_metrics(cls):
        return [cls.BLEU1, cls.BLEU2, cls.BLEU3,
                cls.GLEU1, cls.GLEU2, cls.GLEU3,
                cls.PRECISION, cls.RECALL, cls.F1,
                cls.MICRO_PRECISION,
                cls.MICRO_RECALL,
                cls.MICRO_F1]

    @classmethod
    def get_bleu_eval_metrics(cls):
        return [cls.BLEU1, cls.BLEU2, cls.BLEU3]

    @classmethod
    def get_gleu_eval_metrics(cls):
        return [cls.GLEU1, cls.GLEU2, cls.GLEU3]

    @classmethod
    def is_bleu_eval_metric(cls, eval_metric):
        return eval_metric in cls.get_bleu_eval_metrics()

    @classmethod
    def is_gleu_eval_metric(cls, eval_metric):
        return eval_metric in cls.get_gleu_eval_metrics()

    @classmethod
    def is_bleu_or_gleu_eval_metric(cls, eval_metric):
        return cls.is_bleu_eval_metric(eval_metric) or cls.is_gleu_eval_metric(eval_metric)

    @classmethod
    def is_micro_eval_metric(cls, eval_metric):
        return eval_metric in [cls.MICRO_PRECISION, cls.MICRO_RECALL, cls.MICRO_F1]

    @classmethod
    def get_TED_eval_metrics(cls):
        return [cls.MIN_TED, cls.NORM_MIN_TED, cls.AVG_TED, cls.NORM_AVG_TED]


def validate_schema_config(model_dir_path):
    config_file_path = os.path.join(os.path.dirname(model_dir_path), "args.txt")
    with open(config_file_path, "r") as config_file:
        config_line = config_file.readline()
        config_line = config_line.replace("false", "False")
        config_line = config_line.replace("true", "True")
        config_line = config_line.replace("null", "None")
        config_args = eval(config_line)
        if 'schema' in config_args:
            assert cfg.schema == config_args["schema"]
        else:
            assert cfg.schema == 'NETWORKING'


class Evaluator(object):

    def __init__(self, model_dir_path, greedy_path, inter_greedy_path, references_path,
                 inter_model_dir_path,
                 softmax_model_dir_path,
                 softmax_list_model_dir_path,
                 command_args=None, k=10):
        """

        Args:
            model_dir_path: Path to an agent with an paramsoftmax architecture and binning
            greedy_path:
            inter_greedy_path: Path to interestingness greedy agent
            references_path:
            inter_model_dir_path: Path to an agent with an interestingness component only.
            softmax_model_dir_path:  Path to an agent with an softmax architecture and binning for filter
            terms
            softmax_list_model_dir_path:  Path to an agent with an softmax architecture and fixed list for
            filter terms
            command_args:
        """
        self.schema_name = SchemaName(cfg.schema)

        # validate schema configuration
        validate_schema_config(model_dir_path)
        validate_schema_config(inter_model_dir_path)
        validate_schema_config(softmax_model_dir_path)
        validate_schema_config(softmax_list_model_dir_path)

        if self.schema_name is SchemaName.NETWORKING:
            logger.warning(f"Evaluating agent for NETWORKING schema!")
        elif self.schema_name is SchemaName.FLIGHTS:
            logger.warning(f"Evaluating agent for FLIGHTS schema!")
        else:
            raise NotImplementedError
        """
        Create Actions
        """

        self.K = k  # Number of random actions per dataset

        # 1. Create agent actions

        # 1.1 Regular agent (with all reward components)
        self.D = None  # num of datasets (to be determined in get_agent_actions)
        all_types_of_agent_actions = self.get_agent_actions(model_dir_path, command_args, k)
        self.agent_random_actions = all_types_of_agent_actions["agent_random"]
        self.agent_k_random_actions = all_types_of_agent_actions["agent_k_random"]
        self.agent_max_k_random_actions = all_types_of_agent_actions["agent_max_k"]
        self.agent_most_probable_actions = all_types_of_agent_actions["agent_most_probable"]

        # 1.2 Interestingness agent
        all_types_of_inter_agent_actions = self.get_agent_actions(inter_model_dir_path, command_args, k)
        self.inter_agent_most_probable_actions = all_types_of_inter_agent_actions["agent_most_probable"]

        # 1.3 Softmax agent with binning
        all_types_of_softmax_agent_actions = self.get_agent_actions(softmax_model_dir_path, command_args, k)
        self.softmax_agent_most_probable_actions = all_types_of_softmax_agent_actions["agent_most_probable"]

        # 1.4 Softmax agent with fixed filter terms list
        all_types_of_softmax_list_agent_actions = self.get_agent_actions(softmax_list_model_dir_path, command_args, k)
        self.softmax_list_agent_most_probable_actions = all_types_of_softmax_list_agent_actions["agent_most_probable"]

        # 2.1 Create greedy actions
        self.greedy_actions = self.get_greedy_actions(greedy_path)

        # 2.2 Create interestingness greedy actions
        self.inter_greedy_actions = self.get_greedy_actions(inter_greedy_path)

        # 3. Create references actions
        self.refs_actions = self.get_references_actions(references_path)

        """
        Create Tokens for Actions
        """

    @staticmethod
    def set_framework_cfg_to_human():
        cfg.MAX_NUM_OF_STEPS = 12
        cfg.arch = ArchName.FF_GAUSSIAN.value
        cfg.obs_with_step_num = False
        cfg.stack_obs_num = 1

    def get_greedy_actions(self, greedy_path):
        """
        Assumption:
        Directories and files are in the following structure and name:
            "dataset0.txt":
            "dataset1.txt":
            "dataset2.txt":
            "dataset3.txt":

        Args:
            greedy_path:

        Returns:

        """
        greedy_cands = [None for _ in range(self.D)]
        assert os.path.isdir(greedy_path)

        for dataset_number, dataset_path in enumerate(sorted(os.listdir(greedy_path))):
            dataset_path = os.path.join(greedy_path, dataset_path)
            assert os.path.isfile(dataset_path)
            with open(dataset_path, 'r') as session_file:
                session_actions = session_file.read()
                session = eval(session_actions)
                greedy_cands[dataset_number] = session

        return greedy_cands

    def get_references_actions(self, references_path):
        """
        Assumption:
        Directories and files are in the following structure and name:
        <references_path>:
            "dataset0":
                "session0.txt"
                "session1.txt"
                .
                .
                .
            "dataset1"
            "dataset2"
            "dataset3"


        Args:
            references_path:

        Returns:

        """
        referencess = [[] for _ in range(self.D)]
        assert os.path.isdir(references_path)

        for dataset_number, dataset_path in enumerate(sorted(os.listdir(references_path))):
            dataset_path = os.path.join(references_path, dataset_path)
            assert os.path.isdir(dataset_path)
            for session_file_path in sorted(os.listdir(dataset_path)):
                session_file_path = os.path.join(dataset_path, session_file_path)
                with open(session_file_path, 'r') as session_file:
                    session_actions = session_file.read()
                    session = eval(session_actions)
                    referencess[dataset_number].append(session)

        return referencess

    def get_agent_actions(self, model_dir_path, command_args, k):
        # Command line argumnets (add --episode-length, --bound-mean, --stack-obs-num, etc.)
        if command_args is None:
            command_args = '--algo chainerrl_ppo --arch FFParamSoftmax --episode-length 12 --stack-obs-num 3'

        command = "train.py --env ATENAcont-v0 --demo %s --load %s" % (command_args, model_dir_path)
        sys.argv = command.split()
        agent, env, args = initialize_agent_and_env()
        self.D = len(env.env.repo.data)

        # Set environment properties
        gep.update_global_env_prop_from_cfg()

        # Random actions based on the policy of the agent
        agent_random_actions = [get_actions_lst_of_agent_for_dataset(agent, env, i) for
                                i in range(self.D)]

        # k random actions for each dataset based on the random policy of the agent
        agent_k_random_actions = [[] for _ in range(self.D)]
        agent_k_random_rewards = [[] for _ in range(self.D)]
        for dataset_number in range(self.D):
            for _ in range(k):
                actions_lst, total_reward = get_actions_lst_and_total_reward_of_agent_for_dataset(
                    agent, env, dataset_number)
                agent_k_random_actions[dataset_number].append(actions_lst)
                agent_k_random_rewards[dataset_number].append(total_reward)

        # Max out of k random actions for each dataset
        argmax_idxs = [
            np.argmax(agent_k_random_rewards[dataset_number])
            for dataset_number in range(self.D)
        ]
        agent_max_k_random_actions = [
            agent_k_random_actions[dataset_number][argmax_idxs[dataset_number]]
            for dataset_number in range(self.D)
        ]

        # Most probable (deterministic) actions of the agent's policy for each dataset
        agent_most_probable_actions = [get_most_probable_actions_lst_of_agent_for_dataset(agent, env, i) for
                                       i in range(self.D)]

        # restore human config
        Evaluator.set_framework_cfg_to_human()

        result = {
            "agent_random": agent_random_actions,
            "agent_k_random": agent_k_random_actions,
            "agent_max_k": agent_max_k_random_actions,
            "agent_most_probable": agent_most_probable_actions
        }

        return result

    def calculate_tree_bleu_for_displays(self, references_actions, candidate_actions,
                                         compressed_ref=False,
                                         filter_by_field_ref=True,
                                         continuous_filter_term_ref=True,
                                         compressed_cand=False,
                                         filter_by_field_cand=True,
                                         continuous_filter_term_cand=True,
                                         eval_method=None,
                                         is_gleu=False
                                         ):

        references, candidates = get_refs_and_cands_tokens_from_actions(
            references_actions, candidate_actions,
            compressed_ref=compressed_ref,
            filter_by_field_ref=filter_by_field_ref,
            continuous_filter_term_ref=continuous_filter_term_ref,
            compressed_cand=compressed_cand,
            filter_by_field_cand=filter_by_field_cand,
            continuous_filter_term_cand=continuous_filter_term_cand
        )

        # Calculate tree corpus BLEU / GLEU
        chencherry = SmoothingFunction()
        corpus_bleus = []
        for bleu_n in range(1, 4):
            if not is_gleu:
                corpus_bleus.append(tree_corpus_bleu_n(references, candidates, back_token='[back]', n=bleu_n,
                                                       smoothing_function=chencherry.method1))
            elif is_gleu:
                corpus_bleus.append(tree_corpus_gleu_n(references, candidates, back_token='[back]', n=bleu_n))

        # Calculate tree sentence BLEU for each candidate individually
        sentence_bleus = [dict() for _ in range(self.D)]
        for dataset_num in range(self.D):
            if eval_method is None:
                refs, cand = references[dataset_num], candidates[dataset_num]
            for bleu_n in range(1, 4):
                if not is_gleu:
                    sentence_bleus[dataset_num][bleu_n] = tree_sentence_bleu_n(
                        refs, cand, back_token='[back]', n=bleu_n, smoothing_function=chencherry.method1)
                elif is_gleu:
                    sentence_bleus[dataset_num][bleu_n] = tree_sentence_gleu_n(
                        refs, cand, back_token='[back]', n=bleu_n)

        result = {
            'corpus_tree_bleu1': corpus_bleus[0],
            'corpus_tree_bleu2': corpus_bleus[1],
            'corpus_tree_bleu3': corpus_bleus[2],
            'sentence_bleus': sentence_bleus
        }

        return result

    def calculate_eval_metric_for_displays(self, references_actions, candidate_actions,
                                           eval_metric,
                                           compressed_ref=False,
                                           filter_by_field_ref=True,
                                           continuous_filter_term_ref=True,
                                           compressed_cand=False,
                                           filter_by_field_cand=True,
                                           continuous_filter_term_cand=True,
                                           eval_method=None
                                           ):
        if EvalMetric.is_micro_eval_metric(eval_metric):
            get_tokens_func = get_refs_and_multiple_cands_tokens_from_actions
        else:
            get_tokens_func = get_refs_and_cands_tokens_from_actions

        references, candidates = get_tokens_func(
            references_actions, candidate_actions,
            compressed_ref=compressed_ref,
            filter_by_field_ref=filter_by_field_ref,
            continuous_filter_term_ref=continuous_filter_term_ref,
            compressed_cand=compressed_cand,
            filter_by_field_cand=filter_by_field_cand,
            continuous_filter_term_cand=continuous_filter_term_cand
        )

        # Calculate the metric's value for candidate individually
        metric_val_per_dataset = [0 for _ in range(self.D)]
        values_to_avg = []
        for dataset_num in range(self.D):
            if eval_method is None:
                refs, cand = references[dataset_num], candidates[dataset_num]
            metric_val_per_dataset[dataset_num] = Evaluator.calculate_eval_metric(
                refs, cand, eval_metric=eval_metric)
            values_to_avg.append(metric_val_per_dataset[dataset_num])

        result = {
            'avg_metric_vals': np.mean(values_to_avg),
            'metric_val_per_dataset': metric_val_per_dataset
        }

        return result

    def get_ith_k_random_agent_actions(self, i):
        actions = []
        for dataset_number in range(self.D):
            actions.append(self.agent_k_random_actions[dataset_number][i])

        return actions

    def get_actions_for_eval_method(self, eval_method):
        if eval_method is EvalMethod.RANDOM:
            actions = self.agent_random_actions
        elif eval_method is EvalMethod.MAX_K_RANDOM:
            actions = self.agent_k_random_actions
        elif eval_method is EvalMethod.MOST_PROBABLE:
            actions = self.agent_most_probable_actions
        elif eval_method is EvalMethod.INTER_MOST_PROBABLE:
            actions = self.inter_agent_most_probable_actions
        elif eval_method is EvalMethod.SOFTMAX_MOST_PROBABLE:
            actions = self.softmax_agent_most_probable_actions
        elif eval_method is EvalMethod.SOFTMAX_LIST_MOST_PROBABLE:
            actions = self.softmax_list_agent_most_probable_actions
        elif eval_method is EvalMethod.GREEDY:
            actions = self.greedy_actions
        elif eval_method is EvalMethod.INTERESTINGNESS_GREEDY:
            actions = self.inter_greedy_actions
        elif eval_method is EvalMethod.REFERENCE:
            actions = self.refs_actions
        else:
            raise NotImplementedError
        return actions

    def get_actions_for_dataset_num(self, eval_method, dataset_num):
        actions = self.get_actions_for_eval_method(eval_method)
        return actions[dataset_num]

    ###############################
    # Display in notebook helpers #
    ###############################
    def pad_sentences_with_nulls(self, sentences):
        max_length = max([len(sentence) for sentence in sentences])
        return [sentence + (max_length - len(sentence)) * ['None'] for sentence in sentences]

    def display_ref_cand_comparison(self, reference, candidate):
        cols = [f"Ref{i}" for i in range(len(reference))] + ["Candidate"]
        sentences = reference + [candidate]
        sentences = self.pad_sentences_with_nulls(sentences)
        display(pd.DataFrame(list(zip(*sentences)), columns=cols))

    def get_eval_methods(self):
        schema_name = SchemaName(cfg.schema)
        if schema_name is SchemaName.NETWORKING:
            eval_methods = [EvalMethod.MOST_PROBABLE,
                            #EvalMethod.MAX_K_RANDOM, EvalMethod.K_RANDOM, EvalMethod.RANDOM,
                            EvalMethod.INTER_MOST_PROBABLE,
                            EvalMethod.SOFTMAX_MOST_PROBABLE,
                            EvalMethod.SOFTMAX_LIST_MOST_PROBABLE,
                            EvalMethod.GREEDY, EvalMethod.INTERESTINGNESS_GREEDY
                        ]
        elif schema_name is SchemaName.FLIGHTS:
            eval_methods = [EvalMethod.MOST_PROBABLE,
                            # EvalMethod.MAX_K_RANDOM, EvalMethod.K_RANDOM, EvalMethod.RANDOM,
                            EvalMethod.INTER_MOST_PROBABLE,
                            EvalMethod.SOFTMAX_MOST_PROBABLE,
                            EvalMethod.SOFTMAX_LIST_MOST_PROBABLE,
                            EvalMethod.GREEDY, EvalMethod.INTERESTINGNESS_GREEDY
                            ]
        else:
            raise NotImplementedError
        return eval_methods

    def get_eval_methods_strs(self, eval_methods=None):
        if eval_methods is None:
            eval_methods = self.get_eval_methods()
        eval_methods_strs = [eval_method.value for eval_method in eval_methods]
        return eval_methods_strs

    def highlight_max(self, data, color='yellow'):
        """
        highlight the maximum in a Series or DataFrame
        see https://stackoverflow.com/questions/45606458/python-pandas-highlighting-maximum-value-in-column
        https://stackoverflow.com/questions/55688221/highlight-max-min-on-multiindex-dataframe-pandas
        """
        attr = 'background-color: {}'.format(color)
        # remove % and cast to float
        data = data.replace('%', '', regex=True).astype(float)
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            # is_max = data == data.max().max()
            is_max = data == data.groupby(level=0).transform('max')
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)

    def highlight_min(self, data):
        color = 'yellow'
        attr = 'background-color: {}'.format(color)

        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_min = data == data.min()
            return [attr if v else '' for v in is_min]
        else:
            is_min = data.groupby(level=0).transform('min') == data
            return pd.DataFrame(np.where(is_min, attr, ''),
                                index=data.index, columns=data.columns)

    def highlight_max_min(self, data):
        """
        see https://stackoverflow.com/questions/41654949/pandas-style-function-to-hignlight-specific-columns
        """
        color = 'yellow'
        attr = 'background-color: {}'.format(color)

        # copy df to new - original data are not changed
        df = data.copy()
        # select all values to default value - red color

        first_max_cols = 12

        # df_max = df[df['bleu1','bleu2', 'bleu3', 'precision', 'recall', 'f1']]
        df_max = df.iloc[:, :first_max_cols]
        is_max = df_max == df_max.groupby(level=0).transform('max')

        # df_min = df[df['min_TED', 'min_TED_norm', 'avg_TED', 'avg_TED_norm']]
        df_min = df.iloc[:, first_max_cols:]
        is_min = df_min == df_min.groupby(level=0).transform('min')

        df_max_colors = pd.DataFrame(np.where(is_max, attr, ''),
                                     index=df_max.index, columns=df_max.columns)

        df_min_colors = pd.DataFrame(np.where(is_min, attr, ''),
                                     index=df_min.index, columns=df_min.columns)

        return pd.concat([df_max_colors, df_min_colors], axis=1)

        df.loc[:, :] = 'background-color: red'
        # overwrite values grey color
        df[['B', 'C']] = 'background-color: grey'
        # return color df
        return df

    def display_highlighted_max(self, df):
        display(df.style.apply(self.highlight_max, axis=None))

    def display_highlighted_min(self, df):
        display(df.style.apply(self.highlight_min, axis=None))

    def display_highlighted_max_min(self, df):
        display(df.style.apply(self.highlight_max_min, axis=None))

    ###########################
    # Action to token helpers #
    ###########################

    @staticmethod
    def action_idx2token(action_idx, gran_level="action_type"):
        (action_type, parameters) = gep.global_env_prop.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION[action_idx]
        action_type_string = gep.global_env_prop.OPERATOR_TYPE_LOOKUP[action_type]
        if action_type_string == "back":
            attr_string = None
        else:
            attr_string = gep.global_env_prop.env_dataset_prop.KEYS[parameters[0]]
        if action_type_string in {"back", "group"}:
            filter_op_string = None
        else:
            filter_op_string = ATENAUtils.INT_OPERATOR_MAP_ATENA_STR[(parameters[1] + 1) * 3 - 1]

        return Evaluator.to_token_helper(action_type_string, attr_string, filter_op_string, gran_level)

    @staticmethod
    def action_vec2token(action_vec, gran_level="action_type"):
        action_type, attr, filter_op = action_vec[0], action_vec[1], action_vec[2]
        action_type_string = gep.global_env_prop.OPERATOR_TYPE_LOOKUP[action_type]
        attr_string = gep.global_env_prop.env_dataset_prop.KEYS[attr]
        filter_op_string = ATENAUtils.INT_OPERATOR_MAP_ATENA_STR[filter_op]

        return Evaluator.to_token_helper(action_type_string, attr_string, filter_op_string, gran_level)

    @staticmethod
    def to_token_helper(action_type_string,
                        attr_string,
                        filter_op_string,
                        gran_level="action_type"):
        if gran_level == "action_type" or action_type_string == "back":
            return f'[{action_type_string}]'
        elif gran_level == "attribute" or action_type_string == "group":
            return f'[{action_type_string}]_[{attr_string}]'
        elif gran_level == "filter_op":
            # changing filter operator from 'contains' to 'equals' for rows other than info_line
            if filter_op_string == "contains" and attr_string != "info_line":
                filter_op_string = "eq"
            return f'[{action_type_string}]_[{attr_string}]_[{filter_op_string}]'

    def nested_actions_lsts_to_tokens(self, nested_actions_lsts, gran_level='action_type'):
        return [self.actions_lsts_to_tokens(actions_lsts, gran_level=gran_level)
                for actions_lsts in nested_actions_lsts]

    def actions_lsts_to_tokens(self, actions_lsts, gran_level="action_type"):
        return [self.actions_lst_to_tokens_lst(actions_lst, gran_level=gran_level)
                for actions_lst in actions_lsts]

    def actions_lst_to_tokens_lst(self, actions_lst, gran_level="action_type"):
        return [self.action_vec2token(action_vec, gran_level) for action_vec in actions_lst]

    ######################
    # Evaluation Methods #
    ######################

    """
    Tree BLEU for displays
    """

    @lru_cache(maxsize=1)
    def displays_tree_bleu_agent_random(self):
        return self.displays_tree_bleu(EvalMethod.RANDOM)

    @lru_cache()
    def displays_tree_bleu_agent_k_random(self, is_gleu=False):
        """

        Returns: average of k scores for each dataset

        """

        tree_bleus = defaultdict(list)

        for i in range(self.K):
            cands = self.get_ith_k_random_agent_actions(i)
            for key, val in self.calculate_tree_bleu_for_displays(self.refs_actions, cands, is_gleu=is_gleu).items():
                tree_bleus[key].append(val)

        # Calculate average
        result = self.calculate_avg_tree_bleus(tree_bleus)

        return result

    def calculate_avg_tree_bleus(self, tree_bleus):
        result = dict()
        # Corpus
        for key in ['corpus_tree_bleu1', 'corpus_tree_bleu2', 'corpus_tree_bleu3']:
            result[key] = np.mean(tree_bleus[key])

        # Sentence
        sentence_bleus_elements = [defaultdict(list) for _ in range(self.D)]
        for elem in tree_bleus['sentence_bleus']:
            for dataset_num in range(self.D):
                for bleu_n in range(1, 4):
                    sentence_bleus_elements[dataset_num][bleu_n].append(elem[dataset_num][bleu_n])
        sentence_bleus = [dict() for _ in range(self.D)]
        for dataset_num in range(self.D):
            for bleu_n in range(1, 4):
                sentence_bleus[dataset_num][bleu_n] = np.mean(sentence_bleus_elements[dataset_num][bleu_n])
        result['sentence_bleus'] = sentence_bleus
        return result

    @lru_cache(maxsize=1)
    def displays_tree_bleu_agent_max_k_random(self):
        return self.displays_tree_bleu(EvalMethod.MAX_K_RANDOM)

    @lru_cache(maxsize=1)
    def displays_tree_bleu_agent_most_probable(self):
        return self.displays_tree_bleu(EvalMethod.MOST_PROBABLE)

    @lru_cache()
    def displays_tree_bleu(self, eval_method, is_gleu=False):
        if eval_method is EvalMethod.RANDOM:
            cands = self.agent_random_actions
        elif eval_method is EvalMethod.MAX_K_RANDOM:
            cands = self.agent_max_k_random_actions
        elif eval_method is EvalMethod.MOST_PROBABLE:
            cands = self.agent_most_probable_actions
        elif eval_method is EvalMethod.INTER_MOST_PROBABLE:
            cands = self.inter_agent_most_probable_actions
        elif eval_method is EvalMethod.SOFTMAX_MOST_PROBABLE:
            cands = self.softmax_agent_most_probable_actions
        elif eval_method is EvalMethod.SOFTMAX_LIST_MOST_PROBABLE:
            cands = self.softmax_list_agent_most_probable_actions
        elif eval_method is EvalMethod.GREEDY:
            cands = self.greedy_actions
        elif eval_method is EvalMethod.INTERESTINGNESS_GREEDY:
            cands = self.inter_greedy_actions
        elif eval_method is EvalMethod.K_RANDOM:
            return self.displays_tree_bleu_agent_k_random(is_gleu=is_gleu)
        else:
            raise NotImplementedError

        return self.calculate_tree_bleu_for_displays(self.refs_actions, cands, is_gleu=is_gleu)

    def displays_sentence_tree_bleu_n_for_dataset(self, eval_method, eval_metric, dataset_num, is_gleu=False):
        assert EvalMetric.is_bleu_or_gleu_eval_metric(eval_metric)
        bleu_n = EvalMetric.map_bleu_eval_metric_to_num(eval_metric)
        all_bleus = self.displays_tree_bleu(eval_method, is_gleu=is_gleu)
        return all_bleus['sentence_bleus'][dataset_num][bleu_n]

    def displays_corpus_tree_bleu_n(self, eval_method, eval_metric, is_gleu=False
                                    ):
        assert EvalMetric.is_bleu_or_gleu_eval_metric(eval_metric)
        bleu_corpus_str = EvalMetric.map_bleu_eval_metric_to_corpus_str(eval_metric)
        all_bleus = self.displays_tree_bleu(eval_method, is_gleu=is_gleu)
        return all_bleus[bleu_corpus_str]


    @lru_cache(maxsize=50)
    def displays_eval_metric(self, eval_method, eval_metric):
        if eval_method is EvalMethod.RANDOM:
            cands = self.agent_random_actions
        elif eval_method is EvalMethod.MAX_K_RANDOM:
            cands = self.agent_max_k_random_actions
        elif eval_method is EvalMethod.MOST_PROBABLE:
            cands = self.agent_most_probable_actions
        elif eval_method is EvalMethod.INTER_MOST_PROBABLE:
            cands = self.inter_agent_most_probable_actions
        elif eval_method is EvalMethod.SOFTMAX_MOST_PROBABLE:
            cands = self.softmax_agent_most_probable_actions
        elif eval_method is EvalMethod.SOFTMAX_LIST_MOST_PROBABLE:
            cands = self.softmax_list_agent_most_probable_actions
        elif eval_method is EvalMethod.GREEDY:
            cands = self.greedy_actions
        elif eval_method is EvalMethod.INTERESTINGNESS_GREEDY:
            cands = self.inter_greedy_actions
        elif eval_method is EvalMethod.K_RANDOM:
            return self.displays_eval_metric_agent_k_random(eval_metric)
        else:
            raise NotImplementedError

        if EvalMetric.is_micro_eval_metric(eval_metric):
            cands = [cands]
        return self.calculate_eval_metric_for_displays(self.refs_actions, cands, eval_metric)

    def displays_eval_metric_for_dataset(self, eval_method, eval_metric, dataset_num):
        eval_metric_vals = self.displays_eval_metric(eval_method, eval_metric)
        return eval_metric_vals['metric_val_per_dataset'][dataset_num]

    def displays_eval_metric_avg_all_datasets(self, eval_method, eval_metric):
        eval_metric_vals = self.displays_eval_metric(eval_method, eval_metric)
        return eval_metric_vals['avg_metric_vals']

    def displays_eval_metric_multiple_sessions(self, eval_metric,
                                               num_of_sessions_per_dataset,
                                               get_ith_action_in_each_dataset_func):
        """

        Returns: average of scores for each dataset

        """

        if EvalMetric.is_micro_eval_metric(eval_metric):
            cands = []
            for i in range(num_of_sessions_per_dataset):
                cand_i = get_ith_action_in_each_dataset_func(i)
                cands.append(cand_i)

            result = self.calculate_eval_metric_for_displays(
                    self.refs_actions, cands, eval_metric,
                )
        else:
            eval_metric_values = defaultdict(list)

            for i in range(num_of_sessions_per_dataset):
                cands = get_ith_action_in_each_dataset_func(i)
                for key, val in self.calculate_eval_metric_for_displays(
                        self.refs_actions, cands, eval_metric,
                ).items():
                    eval_metric_values[key].append(val)

            # Calculate average
            result = self.calculate_avg_eval_metric(eval_metric_values)

        return result

    def displays_eval_metric_agent_k_random(self, eval_metric):
        """

        Returns: average of scores for each dataset

        """

        return self.displays_eval_metric_multiple_sessions(
            eval_metric,
            self.K,
            self.get_ith_k_random_agent_actions
        )

    def calculate_avg_eval_metric(self, eval_metric_values):
        result = dict()
        for key in ['avg_metric_vals', 'metric_val_per_dataset']:
            result[key] = np.mean(eval_metric_values[key], axis=0)
        return result

    @staticmethod
    def calculate_eval_metric(refs, cand, eval_metric):
        if eval_metric is EvalMetric.PRECISION:
            return precision_score_without_back(refs, cand, back_token='[back]')
        elif eval_metric is EvalMetric.RECALL:
            return recall_score_without_back(refs, cand, back_token='[back]')
        elif eval_metric is EvalMetric.F1:
            return f1_score_without_back(refs, cand, back_token='[back]')
        elif eval_metric is EvalMetric.MICRO_PRECISION:
            return micro_precision_without_back(refs, cand, back_token='[back]')
        elif eval_metric is EvalMetric.MICRO_RECALL:
            return micro_recall_without_back(refs, cand, back_token='[back]')
        elif eval_metric is EvalMetric.MICRO_F1:
            return micro_f1_without_back(refs, cand, back_token='[back]')
        else:
            raise  NotImplementedError

    @lru_cache(maxsize=1)
    def display_tree_bleu_displays_summary(self, display=True):
        columns = ['bleu1', 'bleu2', 'bleu3', 'gleu1', 'gleu2', 'gleu3', 'precision', 'recall', 'f1',
                   'micro-precision', 'micro-recall', 'micro-f1']
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'avg_all_datasets']
        eval_methods = self.get_eval_methods()
        arrays = [np.array([dataset for dataset in datasets for _ in range(len(eval_methods))]),
                  np.array(self.get_eval_methods_strs() * len(datasets))]

        data = []

        # Every dataset individually
        for dataset_num in range(self.D):
            for eval_method in eval_methods:
                data_to_append = []
                for eval_metric in EvalMetric.get_bleu_and_f1_eval_metrics():
                    if eval_metric in EvalMetric.get_bleu_eval_metrics():  # BLEU
                        data_elem = self.displays_sentence_tree_bleu_n_for_dataset(eval_method,
                                                                                   eval_metric,
                                                                                   dataset_num)
                    elif eval_metric in EvalMetric.get_gleu_eval_metrics():  # GLEU
                        data_elem = self.displays_sentence_tree_bleu_n_for_dataset(eval_method,
                                                                                   eval_metric,
                                                                                   dataset_num,
                                                                                   is_gleu=True
                                                                                   )
                    else:  # Non-BLEU
                        data_elem = self.displays_eval_metric_for_dataset(
                            eval_method, eval_metric, dataset_num)
                    data_to_append.append(data_elem)
                data.append(data_to_append)

        # Average of all datasets
        for eval_method in eval_methods:
            data_to_append = []
            for eval_metric in EvalMetric.get_bleu_and_f1_eval_metrics():
                if eval_metric in EvalMetric.get_bleu_eval_metrics():  # BLEU
                    data_elem = self.displays_corpus_tree_bleu_n(eval_method,
                                                                 eval_metric
                                                                 )
                elif eval_metric in EvalMetric.get_gleu_eval_metrics():  # GLEU
                    data_elem = self.displays_corpus_tree_bleu_n(eval_method,
                                                                 eval_metric,
                                                                 is_gleu=True
                                                                 )
                else:  # Non-BLEU
                    data_elem = self.displays_eval_metric_avg_all_datasets(eval_method, eval_metric)
                data_to_append.append(data_elem)
            data.append(data_to_append)

        df_smmary_tree_bleu_for_displays = pd.DataFrame(np.around(data, decimals=2), index=arrays, columns=columns)
        if display:
            self.display_highlighted_max(df_smmary_tree_bleu_for_displays)

        return df_smmary_tree_bleu_for_displays

    """
    Tree edit distance (TED) for displays
    """

    def calculate_TED_for_displays(self, references_actions, candidate_actions,
                                   compressed_ref=False,
                                   filter_by_field_ref=True,
                                   continuous_filter_term_ref=True,
                                   compressed_cand=False,
                                   filter_by_field_cand=True,
                                   continuous_filter_term_cand=True,
                                   eval_method=None
                                   ):

        # Calculate TED for each dataset individually
        result = AllDatasetsTedResult()
        for dataset_num in range(self.D):
            refs, cand = references_actions[dataset_num], candidate_actions[dataset_num]
            # Unnormalized
            min_ted, argmin_ref, teds_lst, min_ted_edit_operations = compute_minimum_display_TED_from_actions(
                refs, cand, dataset_number=dataset_num, normalize=False, return_min_ops=True,
                compressed_ref=compressed_ref,
                filter_by_field_ref=filter_by_field_ref,
                continuous_filter_term_ref=continuous_filter_term_ref,
                compressed_cand=compressed_cand,
                filter_by_field_cand=filter_by_field_cand,
                continuous_filter_term_cand=continuous_filter_term_cand
            )
            unnorm_ted_result = TedResult(min_ted, argmin_ref, teds_lst, min_ted_edit_operations, normalized=False)
            min_ted, argmin_ref, teds_lst, min_ted_edit_operations = compute_minimum_display_TED_from_actions(
                refs, cand, dataset_number=dataset_num, normalize=True, return_min_ops=True,
                compressed_ref=compressed_ref,
                filter_by_field_ref=filter_by_field_ref,
                continuous_filter_term_ref=continuous_filter_term_ref,
                compressed_cand=compressed_cand,
                filter_by_field_cand=filter_by_field_cand,
                continuous_filter_term_cand=continuous_filter_term_cand
            )
            norm_ted_result = TedResult(min_ted, argmin_ref, teds_lst, min_ted_edit_operations, normalized=True)

            result.append(unnorm_ted_result)
            result.append(norm_ted_result)

        return result

    @lru_cache(maxsize=10)
    def displays_TED(self, eval_method):
        if eval_method is EvalMethod.RANDOM:
            cands = self.agent_random_actions
        elif eval_method is EvalMethod.MAX_K_RANDOM:
            cands = self.agent_max_k_random_actions
        elif eval_method is EvalMethod.MOST_PROBABLE:
            cands = self.agent_most_probable_actions
        elif eval_method is EvalMethod.INTER_MOST_PROBABLE:
            cands = self.inter_agent_most_probable_actions
        elif eval_method is EvalMethod.SOFTMAX_MOST_PROBABLE:
            cands = self.softmax_agent_most_probable_actions
        elif eval_method is EvalMethod.SOFTMAX_LIST_MOST_PROBABLE:
            cands = self.softmax_list_agent_most_probable_actions
        elif eval_method is EvalMethod.GREEDY:
            cands = self.greedy_actions
        elif eval_method is EvalMethod.INTERESTINGNESS_GREEDY:
            cands = self.inter_greedy_actions
        elif eval_method is EvalMethod.K_RANDOM:
            return self.displays_TED_agent_k_random()
        else:
            raise NotImplementedError

        return self.calculate_TED_for_displays(self.refs_actions, cands)

    def displays_TED_agent_k_random(self):
        """

        Returns: average of k scores for each dataset

        """

        all_datasets_ted_results = []

        for i in range(self.K):
            cands = self.get_ith_k_random_agent_actions(i)
            all_datasets_ted_result = self.calculate_TED_for_displays(self.refs_actions, cands)
            all_datasets_ted_results.append(all_datasets_ted_result)

        # Calculate average
        result = self.calculate_avg_all_datasets_ted_results(all_datasets_ted_results)

        return result


    def calculate_avg_all_datasets_ted_results(self, all_datasets_ted_results):
        result = AllDatasetsTedResult()

        min_teds = []
        avg_teds = []
        norm_min_teds = []
        norm_avg_teds = []
        for all_datasets_ted_result in all_datasets_ted_results:
            min_teds.append(all_datasets_ted_result.min_teds)
            avg_teds.append(all_datasets_ted_result.avg_teds)
            norm_min_teds.append(all_datasets_ted_result.norm_min_teds)
            norm_avg_teds.append(all_datasets_ted_result.norm_avg_teds)

        # dummy ted_results_lst
        result.ted_results_lst = all_datasets_ted_result.ted_results_lst

        # Compute averages
        # Unnormalized
        result.min_teds = np.mean(min_teds, axis=0)
        result.avg_teds = np.mean(avg_teds, axis=0)

        # Normalized
        result.norm_min_teds = np.mean(norm_min_teds, axis=0)
        result.norm_avg_teds = np.mean(norm_avg_teds, axis=0)

        return result

    def displays_TED_for_dataset(self, eval_method, eval_metric, dataset_num):
        assert eval_metric in EvalMetric.get_TED_eval_metrics()

        all_datasets_ted_result = self.displays_TED(eval_method)
        return all_datasets_ted_result.get_eval_metric_lst(eval_metric)[dataset_num]

    def calculate_avg_TED_all_datasets(self, eval_method, eval_metric):
        assert eval_metric in EvalMetric.get_TED_eval_metrics()

        all_datasets_ted_result = self.displays_TED(eval_method)
        return all_datasets_ted_result.get_average_all_datasets(eval_metric)

    @lru_cache(maxsize=1)
    def display_TED_displays_summary(self, display=True):
        columns = ['min_TED', 'min_TED_norm', 'avg_TED', 'avg_TED_norm']
        datasets = ['dataset0', 'dataset1', 'dataset2', 'dataset3', 'avg_all_datasets']
        eval_methods = self.get_eval_methods()
        arrays = [np.array([dataset for dataset in datasets for _ in range(len(eval_methods))]),
                  np.array(self.get_eval_methods_strs() * len(datasets))]

        data = []

        # Every dataset individually
        for dataset_num in range(self.D):
            for eval_method in eval_methods:
                data_to_append = []
                for eval_metric in EvalMetric.get_TED_eval_metrics():
                    data_elem = self.displays_TED_for_dataset(eval_method, eval_metric, dataset_num)
                    data_to_append.append(data_elem)
                data.append(data_to_append)

        # Average of all datasets
        for eval_method in eval_methods:
            data_to_append = []
            for eval_metric in EvalMetric.get_TED_eval_metrics():
                data_elem = self.calculate_avg_TED_all_datasets(eval_method, eval_metric)
                data_to_append.append(data_elem)
            data.append(data_to_append)

        df_displays_min_ted = pd.DataFrame(np.around(data, decimals=2), index=arrays, columns=columns)
        if display:
            self.display_highlighted_min(df_displays_min_ted)

        return df_displays_min_ted

    def display_full_displays_summary(self, display=True):
        df_smmary_tree_bleu_for_displays = self.display_tree_bleu_displays_summary(display=False)
        df_displays_min_ted = self.display_TED_displays_summary(display=False)
        df_displays_all_comparison = pd.concat([df_smmary_tree_bleu_for_displays, df_displays_min_ted], axis=1)
        if display:
            self.display_highlighted_max_min(df_displays_all_comparison)

        return df_displays_all_comparison

    @lru_cache(maxsize=10)
    def display_pvalue_displays_summary(self, eval_method):
        columns = ['bleu1', 'bleu2', 'bleu3', 'gleu1', 'gleu2', 'gleu3', 'precision', 'recall', 'f1',
                   'micro-precision', 'micro-recall', 'micro-f1',
                   'min_TED', 'min_TED_norm', 'avg_TED', 'avg_TED_norm'
                   ]
        eval_methods = self.get_eval_methods()
        eval_methods.remove(eval_method)
        arrays = [np.array(self.get_eval_methods_strs(eval_methods))]

        data = []

        for compare_eval_method in eval_methods:
            data_to_append = []
            for eval_metric in EvalMetric.get_bleu_and_f1_eval_metrics() + EvalMetric.get_TED_eval_metrics():
                lst1 = []
                lst2 = []
                for dataset_num in range(self.D):
                    if eval_metric in EvalMetric.get_bleu_eval_metrics():  # BLEU
                        data_elem = self.displays_sentence_tree_bleu_n_for_dataset(eval_method,
                                                                                   eval_metric,
                                                                                   dataset_num)
                        compare_data_elem = self.displays_sentence_tree_bleu_n_for_dataset(compare_eval_method,
                                                                                   eval_metric,
                                                                                   dataset_num)

                    elif eval_metric in EvalMetric.get_gleu_eval_metrics():  # GLEU
                        data_elem = self.displays_sentence_tree_bleu_n_for_dataset(eval_method,
                                                                                   eval_metric,
                                                                                   dataset_num,
                                                                                   is_gleu=True
                                                                                   )
                        compare_data_elem = self.displays_sentence_tree_bleu_n_for_dataset(compare_eval_method,
                                                                                   eval_metric,
                                                                                   dataset_num,
                                                                                   is_gleu=True
                                                                                   )

                    elif eval_metric in EvalMetric.get_TED_eval_metrics():  # TED
                        data_elem = self.displays_TED_for_dataset(eval_method, eval_metric, dataset_num)
                        compare_data_elem = self.displays_TED_for_dataset(compare_eval_method, eval_metric, dataset_num)

                    else:  # Precision, Recall, F1
                        data_elem = self.displays_eval_metric_for_dataset(
                            eval_method, eval_metric, dataset_num)
                        compare_data_elem = self.displays_eval_metric_for_dataset(
                            compare_eval_method, eval_metric, dataset_num)
                    lst1.append(data_elem)
                    lst2.append(compare_data_elem)

                t_statistic, p_val = paired_pvalue(lst1, lst2)
                data_to_append.append(p_val)
            data.append(data_to_append)

        df_pval_for_displays = pd.DataFrame(data, index=arrays, columns=columns)
        display(df_pval_for_displays)

        return df_pval_for_displays

