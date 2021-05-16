import sys
import gym

import Utilities.Configuration.config as cfg
from Utilities.Utility_Functions import initialize_agent_and_env
from gym_atena.envs.atena_env_cont import ATENAEnvCont


def run_random(env, agent=None, dataset_number=None):
    info_hist = []
    real_info_hist = []
    env.render()
    env.reset()
    if isinstance(env,ATENAEnvCont):
        s = env.reset(dataset_number=dataset_number)
    elif isinstance(env,gym.wrappers.Monitor):
        s = env.env.env.reset(dataset_number)
    else:
        s = env.env.reset(dataset_number=dataset_number)

    r_sum = 0
    for ep_t in range(cfg.MAX_NUM_OF_STEPS):
        if not agent:
            print("Warning: no agent, only sampling")
            a = env.action_space.sample()         # estimate stochastic action based on policy
        else:
            a = agent.act(s)
        print(a)
        s_, r, done, info = env.step(a) # make step in environment
        info_hist.append((info,r))
        real_info_hist.append(info)
        s=s_
        r_sum+=r
        if done:
            break
    dhist = env.dhist
    ahist = env.ahist
    return info_hist, r_sum, real_info_hist, dhist, ahist


def simulate(info_hist, displays=False):
    '''
    Details about the actions (reward etc.)
    dispays=True will also show the displays (not only the actions)
    '''
    r_sum = 0
    for i, reward in info_hist:

        print(i["action"], reward)
        print(str(i["reward_info"]))
        if displays:
            f, g = i["raw_display"]
            if g is not None:
                print(g)
            else:
                print(f.head(5))
        r_sum += reward
    print("Total Reward:", r_sum)


def main():
    agent, env, args = initialize_agent_and_env(is_test=True)

    info_hist = run_random(env, agent, args.dataset_number)[0]
    simulate(info_hist, True)


if __name__ == '__main__':
    main()
