from functools import lru_cache

from gym_atena.envs.atena_env_cont import ATENAEnvCont


class GreedyAgent(object):
    """
    A greedy agent that takes in each steps the most rewarding action
    """
    def __init__(self,
                 kl_coeff,
                 compaction_coeff,
                 diversity_coeff,
                 humanity_coeff,
                 ):
        self.kl_coeff = kl_coeff
        self.compaction_coeff = compaction_coeff
        self.diversity_coeff = diversity_coeff
        self.humanity_coeff = humanity_coeff
        self.acts_lst = []
        self.step_num = 0

    @lru_cache(maxsize=32)
    def train(self, dataset_number, episode_length, verbose=True):
        """
        "Train" the agent on the given dataset for `epsiode_length` steps
        Args:
            dataset_number:
            episode_length:
            verbose:

        Returns:

        """
        if verbose:
            print(f'Training greedy algorithm on dataset No. {dataset_number} for {episode_length} steps')
        acts_lst, _ = ATENAEnvCont.get_greedy_max_reward_actions_lst(
            dataset_number=dataset_number,
            episode_length=episode_length,
            kl_coeff=self.kl_coeff,
            compaction_coeff=self.compaction_coeff,
            diversity_coeff=self.diversity_coeff,
            humanity_coeff=self.humanity_coeff,
            verbose=True,
        )
        self.acts_lst = acts_lst

    def act(self, dummy=None):
        """
        Returns the next action of the greedy agent. Note that this method should run only after trainig of the agent
        and that the agent does not need a state to choose it next action. It only stores the current step inside the
        session.
        Args:
            dummy:

        Returns:

        """
        try:
            action = self.acts_lst[self.step_num]
            self.step_num += 1
            return action
        except IndexError:
            raise IndexError("Did you forget to train greedy agent with the correct number of steps?")

    def begin_episode(self):
        """
        A function that mush be called at the beginning of each session.
        Returns:

        """
        self.step_num = 0
