# for coniguration files details see https://martin-thoma.com/configuration-files-in-python/

"""A configuration file"""
# import using: import Utilities.Configuration.config as cfg


# num of step in each session
MAX_NUM_OF_STEPS = 12

# output directory relative path
outdir = ""

"""training configuration"""
num_envs = 1

"""model"""
n_hidden_channels = 64
arch = 'FFGaussian'
beta = 1.0

"""data"""
schema = 'NETWORKING'  # or 'NETWORKING', 'FLIGHTS', 'BIG_FLIGHTS', 'WIDE_FLIGHTS', 'WIDE12_FLIGHTS'
# If None a dataset number of the given schema is chosen randomly for each new episode
# If given, this dataset number for the given schema is used for all episodes
dataset_number = None

"""env"""
# back actions are not available (DEPRECATED!)
no_back = False

# observation vector will include the step number if True
obs_with_step_num = False

# number of previous display vectors that will be stacked in the observation vector (including the current display)
stack_obs_num = 1

# filter term bins sizes
bins_sizes = 'CUSTOM_WIDTH'
exponential_sizes_num_of_bins = 35

"""reward"""
# humanity (coherency) reward
use_humans_reward = True
humans_reward_interval = 64
humanity_coeff = 1.0

# diversity reward
no_diversity = False
diversity_coeff = 1.0

# interestingness reward
no_interestingness = False
kl_coeff = 1.0
compaction_coeff = 1.0
count_data_driven = False

"""logging"""
log_interval = 1000

"""optimizations"""
max_nn_tokens = 12
cache_dfs_size = 750
cache_tokenization_size = 10000
cache_distances_size = 750


