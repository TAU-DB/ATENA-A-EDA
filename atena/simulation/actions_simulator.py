from typing import List

from atena.simulation.actions import (
    AbstractAction,
    ActionType,
    FilterAction,
    GroupAction
)
from atena.simulation.dataset import (
    Dataset,
    SchemaName,
    DatasetMeta,
    DatasetName,
)
from atena.simulation.display import DisplayCalculator
from atena.simulation.state import EnvStateTuple, FilteringTuple, AggregationTuple
from atena.simulation.utils import random_action_generator


class SimulationState(object):
    def __init__(self, display_calculator: DisplayCalculator):
        empty_state = EnvStateTuple.create_empty_state()

        # states_history is a list of all states during simulation
        self.states_history = [empty_state]

        # states_stack is a stack of states during simulation (back actions pop elements from list!)
        self.states_stack = [empty_state]

        # Calculate the initial display
        disp, _ = display_calculator.calculate_display(empty_state)

        # displays_history is a list of all displays during simulation
        self.displays_history = [disp]


class ActionsSimulator(object):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.display_calculator = DisplayCalculator(self.dataset)
        self.simulation_state = SimulationState(self.display_calculator)
        self.is_reset = True

    def reset(self):
        """
        Reset internal state of the simulator
        Returns:

        """
        self.display_calculator = DisplayCalculator(self.dataset)
        self.simulation_state = SimulationState(self.display_calculator)

    def run_actions(self, actions_lst: List[AbstractAction]):
        assert self.is_reset, "Oh no, you should first reset() the simulator!"
        steps_info = []
        for action in actions_lst:
            step_info = self.execute_action(action)
            steps_info.append(step_info)

        self.is_reset = False
        return steps_info

    def execute_action(self, action: AbstractAction):
        """This function processes an action:
         (1) Executes the action: It computes a rolling "state" dictionary, comprising filtering,grouping and aggregations
         (2) Calculates the display vector and new DataFrames
         (3) Update the history lists
        """

        # (1) Executing an action by incrementing the state dictionary:
        if action.action_type is ActionType.BACK:
            # If back: pop the last element from the history and use it as the current state
            if len(self.simulation_state.states_stack) > 1:
                self.simulation_state.states_stack.pop()
                new_state = self.simulation_state.states_stack[-1]
            else:
                new_state = EnvStateTuple.create_empty_state()

        elif action.action_type is ActionType.FILTER:
            assert isinstance(action, FilterAction)
            # If filter: add the filter condition to the list of filters in the prev state
            filt_tpl = FilteringTuple(
                column=action.filtered_column,
                term=action.filter_term,
                operator=action.filter_operator
            )
            new_state = self.simulation_state.states_stack[-1]
            new_state = new_state.append_filtering(filt_tpl)
            self.simulation_state.states_stack.append(new_state)

        elif action.action_type is ActionType.GROUP:
            assert isinstance(action, GroupAction)
            # Add to the grouping and aggregations lists of the prev state:
            new_state = self.simulation_state.states_stack[-1]
            if action.grouped_column not in new_state["grouping"]:
                new_state = new_state.append_grouping(action.grouped_column)
            agg_tpl = AggregationTuple(
                column=action.aggregated_column,
                aggregation_function=action.aggregation_function
            )
            if agg_tpl not in new_state["aggregations"]:
                new_state = new_state.append_aggregations(agg_tpl)
            self.simulation_state.states_stack.append(new_state)
        else:
            raise Exception(f"Unknown action type: {action.action_type}")

        # (2) Calculate display vector and new DataFrames
        display, dfs = self.display_calculator.calculate_display(new_state)

        # (3) Update the history lists:
        self.simulation_state.states_history.append(new_state)
        self.simulation_state.displays_history.append(display)

        return {"action": action,
                "display": display,
                "raw_display": dfs,
                "state": new_state,
                }

    def run_n_random_actions(self, n: int):
        actions_lst = [random_action_generator(self.dataset) for _ in range(n)]
        return self.run_actions(actions_lst)

    @classmethod
    def factory_from_schema_and_dataset_name(
            cls, schema_name: SchemaName,
            dataset_name: DatasetName
    ):
        dataset_meta = DatasetMeta(schema_name, dataset_name)
        dataset = Dataset(dataset_meta)
        actions_simulator = cls(dataset)

        return actions_simulator

    @classmethod
    def get_end_of_simulation_state(
            cls,
            schema_name: SchemaName,
            dataset_name: DatasetName,
            actions_lst: List[AbstractAction]
    ):
        actions_simulator = cls.factory_from_schema_and_dataset_name(schema_name, dataset_name)
        actions_simulator.run_actions(actions_lst)

        return actions_simulator.simulation_state
