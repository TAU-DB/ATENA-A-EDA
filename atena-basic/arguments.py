import argparse
import logging
import sys
from enum import Enum


class AlgoName(Enum):
    CAPG_PPO = 'capg_ppo'
    CAPG_TRPO = 'capg_trpo'
    CHAINERRL_PPO = 'chainerrl_ppo'


class ArchName(Enum):
    FF_GAUSSIAN = 'FFGaussian'
    FF_SOFTMAX = 'FFSoftmax'
    FF_MELLOWMAX = 'FFMellowmax'
    FF_PARAM_SOFTMAX = 'FFParamSoftmax'
    GREEDY = 'GREEDY'

class SchemaName(Enum):
    NETWORKING = 'NETWORKING'
    FLIGHTS = 'FLIGHTS'
    BIG_FLIGHTS = 'BIG_FLIGHTS'
    WIDE_FLIGHTS = 'WIDE_FLIGHTS'
    WIDE12_FLIGHTS = 'WIDE12_FLIGHTS'


class FilterTermsBinsSizes(Enum):
    """
    The way sized of bins to each filter terms are mapped
    """
    EQUAL_WIDTH = 'EQUAL_WIDTH'
    CUSTOM_WIDTH = 'CUSTOM_WIDTH'
    EXPONENTIAL = 'EXPONENTIAL'  # Bins' sizes grows exponentially


algo_names = [e.value for e in AlgoName]
arch_names = [e.value for e in ArchName]
schema_names = [e.value for e in SchemaName]
filter_terms_bins_sizes_names = [e.value for e in FilterTermsBinsSizes]


def get_args(is_test):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v1',
                        help='Gym Env ID', required=True)
    parser.add_argument('--load', type=str, default='', required=is_test,
                        help='Directory path to load a saved agent data from'
                             ' if it is a non-empty string.')

    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device ID. Set to -1 to use CPUs only.')
    parser.add_argument('--algo', default='capg_ppo', choices=algo_names,
                        help='algorithm to use')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--arch', type=str, default='FFGaussian',
                        choices=arch_names)
    parser.add_argument('--schema', type=str, default='NETWORKING',
                        choices=schema_names)
    parser.add_argument('--dataset-number', type=int, default=None,
                        help='A dataset number for the given schema. If None, a dataset'
                             ' number of the given schema is chosen randomly for each new episode')
    parser.add_argument('--bound-mean', action='store_true')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total time steps for training.')
    parser.add_argument('--episode-length', type=int, default=10,
                        help='Number of steps in each episode.')
    parser.add_argument('--eval-interval', type=int, default=100000,
                        help='Interval between evaluation phases in steps.')
    parser.add_argument('--eval-n-runs', type=int, default=100,
                        help='Number of episodes ran in an evaluation phase')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--standardize-advantages', action='store_true')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the env')
    parser.add_argument('--use-humans-reward', action='store_true', default=False,
                        help='Use the third reward type, based on human sessions')
    parser.add_argument('--count-data-driven', action='store_true', default=False,
                        help='Count data driven reward as part of the interestingness component instead of'
                             ' the coherency component')

    parser.add_argument('--humans-reward-interval', type=int, default=64,
                        help='Number of episodes between reevaluation of agent performance')
    parser.add_argument('--no-diversity', action='store_true', default=False,
                        help='Do not use the diversity reward')
    parser.add_argument('--no-inter', action='store_true', default=False,
                        help='Do not use the interestingness reward')
    parser.add_argument('--no-back', action='store_true', default=False,
                        help='Back actions are not available')
    parser.add_argument('--bins-sizes', type=str, default='CUSTOM_WIDTH',
                        choices=filter_terms_bins_sizes_names)
    parser.add_argument('--humanity-coeff', type=float, default=1.0)
    parser.add_argument('--diversity-coeff', type=float, default=2.0)
    parser.add_argument('--kl-coeff', type=float, default=1.5)
    parser.add_argument('--compaction-coeff', type=float, default=2.0)
    parser.add_argument('--offset-steps', type=int, default=0,
                        help='Number of steps to offset training.')
    parser.add_argument('--trpo-update-interval', type=int, default=5000,
                        help='Interval steps of TRPO iterations.')
    parser.add_argument('--ppo-update-interval', type=int, default=2048,
                        help='Interval steps of PPO iterations.')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    parser.add_argument('--use-clipped-gaussian', action='store_true',
                        help='Use ClippedGaussian instead of Gaussian')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=1000)

    parser.add_argument('--adam-lr', type=float, default=3e-4)
    parser.add_argument('--ppo-gamma', type=float, default=0.995)
    parser.add_argument('--ppo-lambda', type=float, default=0.97)
    parser.add_argument('--n-hidden-channels', type=int, default=64,
                        help='Number of hidden channels.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='inverse of the temperature parameter of softmax distribution.')

    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--entropy-coef', type=float, default=0.0)
    parser.add_argument('--label', type=str, default='')

    parser.add_argument('--max-nn-tokens', type=int, default=-1,
                        help='Maximum numbers of nearest neighbors tokens (with the same distance) for choosing'
                             'a filter terms. This is reasonable in training phase only. -1 is used for no boundary ')
    parser.add_argument('--obs-with-step-num', action='store_true', default=False,
                        help='Add the current step number to the observation vector')
    parser.add_argument('--stack-obs-num', type=int, default=1,
                        help='Stack this number of display vectors to the observation vector')
    parser.add_argument('--cache-dfs-size', type=int, default=-1,
                        help='Maximum number of ((dataset_number,state), dfs) pairs to cache. -1 is used for no cache')
    parser.add_argument('--cache-tokenization-size', type=int, default=-1,
                        help='Maximum number of ((dataset_number, state, column), tokenization) pairs to cache.'
                             ' -1 is used for no cache'
                        )
    parser.add_argument('--cache-distances-size', type=int, default=-1,
                        help='Maximum number of ((dataset_number, state1, state2), distance) pairs to cache.'
                             ' -1 is used for no cache'
                        )
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args
