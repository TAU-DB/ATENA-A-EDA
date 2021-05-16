import logging
import os
import pickle
from copy import deepcopy

import chainerrl

from arguments import get_args, AlgoName, FilterTermsBinsSizes
from envs import make_env
from models.chinerrl_models.chainerrl_ppo import PPOchianerrl
from models.clipped_gaussian.train_ppo_gym import PPOModel
from models.clipped_gaussian.train_trpo_gym import TRPOModel

import Utilities.Configuration.config as cfg
from gym_atena.envs.atena_env_cont import ATENAEnvCont


def initialize_agent_and_env(is_test=False):
    """
    Read and parse commandline arguments to the args variable.
    Initiate an agent and environment based on the arguments.

    :return: agent, env, args
    """
    args = get_args(is_test)

    # set schema
    cfg.schema = args.schema

    # set dataset nubmer
    cfg.dataset_number = args.dataset_number

    # set number of steps in session
    cfg.MAX_NUM_OF_STEPS = args.episode_length
    # hack to change the default value of the max_steps argument in the __init__ of ATENAEnvCont to cfg.MAX_NUM_OF_STEPS
    atena_init_default_args = list(ATENAEnvCont.__init__.__defaults__)
    atena_init_default_args[0] = cfg.MAX_NUM_OF_STEPS
    ATENAEnvCont.__init__.__defaults__ = tuple(atena_init_default_args)

    # set env settings
    cfg.stack_obs_num = args.stack_obs_num
    cfg.obs_with_step_num = args.obs_with_step_num
    cfg.no_back = args.no_back
    cfg.bins_sizes = args.bins_sizes
    #filter_terms_bins_sizes_helper(FilterTermsBinsSizes(cfg.bins_sizes))
    #paremetric_softmax_idx_action_maps_helper()

    # set reward types to use
    cfg.no_diversity = args.no_diversity
    cfg.no_interestingness = args.no_inter
    cfg.use_humans_reward = args.use_humans_reward
    cfg.humans_reward_interval = args.humans_reward_interval
    cfg.count_data_driven = args.count_data_driven

    # set number of hidden units for gaussian policy
    cfg.n_hidden_channels = args.n_hidden_channels

    # set architecture type
    cfg.arch = args.arch
    cfg.beta = args.beta

    # optimization settings
    cfg.max_nn_tokens = args.max_nn_tokens
    cfg.cache_dfs_size = args.cache_dfs_size
    cfg.cache_tokenization_size = args.cache_tokenization_size
    cfg.cache_distances_size = args.cache_distances_size

    # set reward coefficients
    cfg.humanity_coeff = args.humanity_coeff
    cfg.diversity_coeff = args.diversity_coeff
    cfg.kl_coeff = args.kl_coeff
    cfg.compaction_coeff = args.compaction_coeff

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)
    cfg.outdir = args.outdir

    # https://stackoverflow.com/questions/13479295/python-using-basicconfig-method-to-log-to-console-and-file
    # logging file path
    log_path = os.path.join(args.outdir, 'training_results.log')
    # set up logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        filename=log_path,
        datefmt='%H:%M:%S'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(args.logger_level)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # set logging of the entire episode every LOG_INTERVAL steps
    cfg.log_interval = args.log_interval
    cfg.num_envs = args.num_envs
    ATENAEnvCont.LOG_INTERVAL = int(args.log_interval / args.num_envs)

    # TODO (baelo): delete it
    # Set filter term bins
    #filter_terms_bins_sizes_helper(FilterTermsBinsSizes(cfg.bins_sizes))
    #paremetric_softmax_idx_action_maps_helper()

    # Set random seed
    chainerrl.misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # create environment
    env = make_env(args, args.env, args.seed, args.render, args.outdir)

    # choose algorithm
    args.algo = AlgoName(args.algo)
    if args.algo is AlgoName.CAPG_PPO:  # capg
        model = PPOModel(env,
                         args.gpu,
                         args.n_hidden_channels,
                         args.adam_lr,
                         args.ppo_update_interval,
                         args.outdir,
                         args.load,
                         args.use_clipped_gaussian)
    elif args.algo is AlgoName.CAPG_TRPO:  # capg
        model = TRPOModel(env,
                          args.gpu,
                          args.n_hidden_channels,
                          args.trpo_update_interval,
                          args.outdir,
                          args.load,
                          args.use_clipped_gaussian)
    elif args.algo == AlgoName.CHAINERRL_PPO:
        model = PPOchianerrl(args, env)
    else:
        raise NotImplementedError

    agent = model.agent

    return agent, env, args


def load_human_session_actions_clusters():
    """
    Load observation-actions_lst pairs of human sessions from a .pickle file
    Note: filters "easy" actions

    :return: a dict of observation-actions_lst pairs
    """
    result_human_displays_actions_clusters_obs_based = dict()

    # load human sessions actions clusters from pickle(deserialize)
    with open('human_actions_clusters.pickle', 'rb') as handle:
        # key is the current observation (a tuple of length observation vector)
        # value is a list of action vectors taken for this observation
        human_displays_actions_clusters_obs_based = pickle.load(handle)

    # leave only difficult observations
    # with open('failures_obs.pickle', 'rb') as handle:
    #     failures_obs = pickle.load(handle)
    #     human_displays_actions_clusters_obs_based = {key: val for key, val in
    #     human_displays_actions_clusters_obs_based.items() if key
    #                                    in failures_obs}
    for obs, action_lst in human_displays_actions_clusters_obs_based.items():
        # remove `back` actions if there are other actions
        is_back_in_act_lst = [act[0] == 0 for act in action_lst]
        if not all(is_back_in_act_lst):
            action_lst = [act for i, act in enumerate(action_lst) if not is_back_in_act_lst[i]]
        result_human_displays_actions_clusters_obs_based[obs] = action_lst

        # remove all observations that have `back` as the only action
        if all(is_back_in_act_lst):
            result_human_displays_actions_clusters_obs_based.pop(obs)

    if cfg.no_back:
        result_human_displays_actions_clusters_obs_based_copy = deepcopy(result_human_displays_actions_clusters_obs_based)
        result_human_displays_actions_clusters_obs_based = dict()
        for obs, action_lst in result_human_displays_actions_clusters_obs_based_copy.items():
            new_action_lst = []
            for act in action_lst:
                if act[0] == 0:
                    raise ValueError("--no-back command-line argument is used but human sessions contains back actions")
                elif act[0] == 1:  # if filter
                    act[0] = 0
                elif act[0] == 2:  # if group
                    act[0] = 1
                new_action_lst.append(act)
            result_human_displays_actions_clusters_obs_based[obs] = new_action_lst

    return result_human_displays_actions_clusters_obs_based


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    # https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
