from abc import ABC, abstractmethod
from typing import List, Optional

from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import pandas as pd

from atena.evaluation.utils import (
    precision_score_without_back,
    compute_minimum_display_TED_from_actions, tree_corpus_bleu_n,
)
from atena.simulation.actions import AbstractAction, ActionType
from atena.simulation.actions_simulator import ActionsSimulator
from atena.simulation.dataset import (
    DatasetMeta,
    SchemaName,
    CyberDatasetName,
    DatasetName,
    FlightsDatasetName,
)
from atena.evaluation.references.cyber.dataset1 import cyber1_references
from atena.evaluation.references.cyber.dataset2 import cyber2_references
from atena.evaluation.references.cyber.dataset3 import cyber3_references
from atena.evaluation.references.cyber.dataset4 import cyber4_references
from atena.evaluation.references.flights.dataset1 import flights1_references
from atena.evaluation.references.flights.dataset2 import flights2_references
from atena.evaluation.references.flights.dataset3 import flights3_references
from atena.evaluation.references.flights.dataset4 import flights4_references


class EvalInstance(object):
    def __init__(self, dataset_meta: DatasetMeta, actions_lst: List[AbstractAction]):
        self.dataset_meta = dataset_meta
        self.actions_lst = actions_lst

    @property
    def references_actions(self):
        schema = self.dataset_meta.schema
        dataset_name = self.dataset_meta.dataset_name

        if schema is SchemaName.CYBER:
            if dataset_name is CyberDatasetName.DATASET1:
                return cyber1_references
            elif dataset_name is CyberDatasetName.DATASET2:
                return cyber2_references
            elif dataset_name is CyberDatasetName.DATASET3:
                return cyber3_references
            elif dataset_name is CyberDatasetName.DATASET4:
                return cyber4_references
            else:
                raise NotImplementedError
        elif schema is SchemaName.FLIGHTS:
            if dataset_name is FlightsDatasetName.DATASET1:
                return flights1_references
            elif dataset_name is FlightsDatasetName.DATASET2:
                return flights2_references
            elif dataset_name is FlightsDatasetName.DATASET3:
                return flights3_references
            elif dataset_name is FlightsDatasetName.DATASET4:
                return flights4_references
        else:
            raise NotImplementedError


class TokensReferencesCandidatePair(object):
    def __init__(self, references_tokens: List[List[str]], candidate_tokens: List[str]):
        self.references_tokens = references_tokens
        self.candidate_tokens = candidate_tokens


class AbstractMetric(ABC):
    displays_token_map = dict()

    def __init__(self, metric_name: str, eval_instances: List[EvalInstance]):
        self.name = metric_name
        self.eval_instances = eval_instances

    @classmethod
    def get_display_tokens_from_actions(
            cls,
            schema_name: SchemaName,
            dataset_name: DatasetName,
            actions_lst: List[AbstractAction],
    ) -> List[Optional[str]]:
        back_token = '[back]'

        # Get displays from actions
        end_of_simulation_state = ActionsSimulator.get_end_of_simulation_state(
            schema_name,
            dataset_name,
            actions_lst
        )
        displays_history = end_of_simulation_state.displays_history

        # Make displays_history and actions equal in length
        actions = [None] + actions_lst

        tokens = []
        for disp, action in zip(displays_history, actions):
            disp = str(disp)
            # Create token
            if disp not in cls.displays_token_map:
                cls.displays_token_map[disp] = len(cls.displays_token_map)

            # Add token to list
            action_type = None if action is None else action.action_type
            if action_type is ActionType.BACK:
                tokens.append(back_token)
            else:
                tokens.append(cls.displays_token_map[disp])

        return tokens

    @property
    def tokens_references_candidate_pairs(self) -> List[TokensReferencesCandidatePair]:
        pairs = []
        for eval_instance in self.eval_instances:
            # References tokens
            references_tokens = []
            for actions_lst in eval_instance.references_actions:
                tokens = self.get_display_tokens_from_actions(
                    eval_instance.dataset_meta.schema, eval_instance.dataset_meta.dataset_name, actions_lst)
                references_tokens.append(tokens)

            # Candidate tokens
            candidate_tokens = self.get_display_tokens_from_actions(
                eval_instance.dataset_meta.schema, eval_instance.dataset_meta.dataset_name, eval_instance.actions_lst)

            pairs.append(TokensReferencesCandidatePair(references_tokens, candidate_tokens))

        return pairs

    @abstractmethod
    def compute(self) -> float:
        """
        Computes the value of this metric
        """
        raise NotImplementedError


class DisplaysTreeBleuMetric(AbstractMetric):
    def __init__(self, bleu_n: int, eval_instances: List[EvalInstance]):
        super().__init__(metric_name=f'T-BLEU-{bleu_n}', eval_instances=eval_instances)
        self.bleu_n = bleu_n

    def compute(self) -> float:
        chencherry = SmoothingFunction()
        tokens_references_candidate_pairs = self.tokens_references_candidate_pairs

        return tree_corpus_bleu_n(
            [pair.references_tokens for pair in tokens_references_candidate_pairs],
            [pair.candidate_tokens for pair in tokens_references_candidate_pairs],
            back_token='[back]',
            n=self.bleu_n,
            smoothing_function=chencherry.method1
        )


class PrecisionMetric(AbstractMetric):
    def __init__(self, eval_instances: List[EvalInstance]):
        super().__init__(metric_name='Precision', eval_instances=eval_instances)

    def compute(self) -> float:
        precisions = [precision_score_without_back(
            pair.references_tokens,
            pair.candidate_tokens,
            back_token='[back]'
        ) for pair in self.tokens_references_candidate_pairs]

        return np.mean(precisions, axis=0).item()


class NormalizedDisplaysEdaSimMetric(AbstractMetric):
    def __init__(self, eval_instances: List[EvalInstance]):
        super().__init__(metric_name='EDA-Sim', eval_instances=eval_instances)

    def compute(self) -> float:
        eda_sim_scores = []
        for eval_instance in self.eval_instances:
            min_ted, _, _, _ = compute_minimum_display_TED_from_actions(
                eval_instance.references_actions,
                eval_instance.actions_lst,
                schema_name=eval_instance.dataset_meta.schema,
                dataset_name=eval_instance.dataset_meta.dataset_name,
                normalize=True,
                return_min_ops=True,
            )

            # Tree Edit Distance is a distance measure and we would like it to be a similarity measure
            eda_sim_scores.append(1 - min_ted)

        return np.mean(eda_sim_scores, axis=0).item()


def get_all_eval_metrics(eval_instances: List[EvalInstance]) -> List[AbstractMetric]:
    eval_metrics = [
        PrecisionMetric(eval_instances),
        DisplaysTreeBleuMetric(1, eval_instances),
        DisplaysTreeBleuMetric(2, eval_instances),
        DisplaysTreeBleuMetric(3, eval_instances),
        NormalizedDisplaysEdaSimMetric(eval_instances)
    ]

    return eval_metrics


def get_dataframe_for_eval_metrics(eval_metrics: List[AbstractMetric]) -> pd.DataFrame:
    columns = [eval_metric.name for eval_metric in eval_metrics]
    eval_scores = [[eval_metric.compute() for eval_metric in eval_metrics]]

    return pd.DataFrame(eval_scores, columns=columns)


def get_dataframe_all_eval_metrics(eval_instances: List[EvalInstance]) -> pd.DataFrame:
    return get_dataframe_for_eval_metrics(get_all_eval_metrics(eval_instances))
