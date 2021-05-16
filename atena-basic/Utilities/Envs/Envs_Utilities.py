from gym_atena.envs.atena_env_cont import ATENAEnvCont
from arguments import ArchName
import Utilities.Configuration.config as cfg


def decompress_and_discretize_actions(agent_actions, env_prop=None):
    """
    DEPRECATED!!!!
    :param agent_actions: a list of raw agent actions (compressed and continuous)
    :return: a list of the same actions non-compressed and discretized (except for filter term)
    """
    agent_discrete_actions = []
    arch = ArchName(cfg.arch)
    for action in agent_actions:
        if arch is ArchName.FF_GAUSSIAN:
            action = env_prop.compressed2full_range(action)
        if arch is ArchName.FF_PARAM_SOFTMAX:
            action = ATENAEnvCont.static_param_softmax_idx_to_action(action)
        filter_term = action[3]
        action_disc = ATENAEnvCont.cont2dis(action)
        action_disc[3] = filter_term
        agent_discrete_actions.append(action_disc)
    return agent_discrete_actions
