#from collections import namedtuple
from Utilities.Utility_Functions import initialize_agent_and_env, load_human_session_actions_clusters
from arguments import AlgoName

from train_agent_chainerrl import train_agent_with_evaluation, train_agent_batch_with_evaluation
import chainerrl
from chainerrl import experiments

from envs import make_env, make_batch_env


def train_or_evaluate(
        args,
        agent,
        env_id,
        env,
        seed,
        render,
        eval_n_runs,
        steps,
        eval_interval,
        outdir,
):
    timestep_limit = env.spec.max_episode_steps

    if args.algo in [AlgoName.CAPG_PPO, AlgoName.CAPG_TRPO]:
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=make_env(args, env_id, seed, render, outdir, is_test=True),
            outdir=outdir,
            steps=steps,
            eval_n_runs=eval_n_runs,
            eval_interval=eval_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=True,
        )
    elif args.algo is AlgoName.CHAINERRL_PPO:
        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.alpha = value

        lr_decay_hook = experiments.LinearInterpolationHook(
            args.steps, args.adam_lr, 0, lr_setter)

        # Linearly decay the clipping parameter to zero
        def clip_eps_setter(env, agent, value):
            agent.clip_eps = value

        clip_eps_decay_hook = experiments.LinearInterpolationHook(
            args.steps, 0.2, 0, clip_eps_setter)

        # load human sessions action clusters
        human_displays_actions_clusters = load_human_session_actions_clusters()

        train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(args, False),
            eval_env=make_batch_env(args, True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            return_window_size=args.window_size,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=True,
            step_hooks=[
                lr_decay_hook,
                clip_eps_decay_hook,
            ],
            use_humans_reward=args.use_humans_reward,
            human_displays_actions_clusters=human_displays_actions_clusters,
            humans_reward_interval=args.humans_reward_interval,
            step_offset=args.offset_steps,
        )


if __name__ == '__main__':
    agent, env, args = initialize_agent_and_env()
    # train model
    train_or_evaluate(args, agent,
                      args.env,
                      env,
                      args.seed,
                      args.render,
                      args.eval_n_runs,
                      args.steps,
                      args.eval_interval,
                      args.outdir)
