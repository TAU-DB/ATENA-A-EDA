import os

import chainer
from chainer import functions as F
import gym
import gym.wrappers
import numpy as np

import chainerrl

from models.clipped_gaussian.clipped_model import ClippedModel
from models.clipped_gaussian.train_trpo_gym import ClippedGaussianPolicy


class ObsNormalizedModel(chainerrl.agents.a3c.A3CSeparateModel):
    """An example of A3C feedforward Gaussian policy."""

    def __init__(self, policy, vf, obs_size):
        super().__init__(policy, vf)
        with self.init_scope():
            self.obs_filter = chainerrl.links.EmpiricalNormalization(
                shape=obs_size
            )

    def __call__(self, obs):
        obs = F.clip(self.obs_filter(obs, update=False),
                     -5.0, 5.0)
        return super().__call__(obs)


class PPOModel(ClippedModel):
    def __init__(
            self,
            env,
            gpu,
            n_hidden_channels,
            adam_lr,
            ppo_update_interval,
            outdir,
            load,
            use_clipped_gaussian=True
    ):
        super().__init__()
        obs_space = env.observation_space
        action_space = env.action_space
        print('Observation space:', obs_space)
        print('Action space:', action_space)

        if not isinstance(obs_space, gym.spaces.Box):
            print("""\
    This example only supports gym.spaces.Box observation spaces. To apply it to
    other observation spaces, use a custom phi function that convert an observation
    to numpy.ndarray of numpy.float32.""")  # NOQA
            return

        # Parameterize log std
        def var_func(x): return F.exp(x) ** 2

        assert isinstance(action_space, gym.spaces.Box)
        # Use a Gaussian policy for continuous action spaces
        if use_clipped_gaussian:
            policy = \
                ClippedGaussianPolicy(
                    obs_space.low.size,
                    action_space.low.size,
                    n_hidden_channels=n_hidden_channels,
                    n_hidden_layers=2,
                    mean_wscale=0.01,
                    nonlinearity=F.tanh,
                    var_type='diagonal',
                    var_func=var_func,
                    var_param_init=0,  # log std = 0 => std = 1
                    min_action=action_space.low.astype(np.float32),
                    max_action=action_space.high.astype(np.float32),
                )
        else:
            policy = \
                chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance(
                    obs_space.low.size,
                    action_space.low.size,
                    n_hidden_channels=n_hidden_channels,
                    n_hidden_layers=2,
                    mean_wscale=0.01,
                    nonlinearity=F.tanh,
                    var_type='diagonal',
                    var_func=var_func,
                    var_param_init=0,  # log std = 0 => std = 1
                )

        # Use a value function to reduce variance
        vf = chainerrl.v_functions.FCVFunction(
            obs_space.low.size,
            n_hidden_channels=n_hidden_channels,
            n_hidden_layers=2,
            last_wscale=0.01,
            nonlinearity=F.tanh,
        )

        model = ObsNormalizedModel(policy, vf, obs_space.low.size)

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            model.to_gpu(gpu)

        opt = chainer.optimizers.Adam(adam_lr)
        opt.setup(model)

        # Draw the computational graph and save it in the output directory.
        fake_obs = chainer.Variable(
            policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
            name='observation')
        chainerrl.misc.draw_computational_graph(
            [model(fake_obs)], os.path.join(outdir, 'model'))

        # Hyperparameters in http://arxiv.org/abs/1709.06560
        agent = chainerrl.agents.PPO(
            model=model,
            optimizer=opt,
            phi=lambda x: x.astype(np.float32, copy=False),
            update_interval=ppo_update_interval,
            gamma=0.995,
            lambd=0.97,
            standardize_advantages=True,
            entropy_coef=0,
        )

        if load:
            agent.load(load)

        self._agent = agent

