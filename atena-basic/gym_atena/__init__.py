import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ATENAld-v0',
    entry_point='gym_ianna.envs:ATENAEnv',
#    timestep_limit=1000,
#    reward_threshold=1.0,
#    nondeterministic = True,
)

register(
    id='ATENAcont-v0',
    entry_point='gym_atena.envs.atena_env_cont:ATENAEnvCont',
    #entry_point='gym_atena.envs:ATENAEnvCont',
#    timestep_limit=1000,
#    reward_threshold=1.0,
#    nondeterministic = True,
)