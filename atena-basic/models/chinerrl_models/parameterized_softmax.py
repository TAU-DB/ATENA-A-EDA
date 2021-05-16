import unittest

import numpy as np

import chainer
from cached_property import cached_property
from chainer import functions as F
from chainerrl import distribution
from chainerrl.distribution import CategoricalDistribution
from chainerrl.links import MLP
from chainerrl.policy import Policy


class ParamSoftmaxDistribution(CategoricalDistribution):
    """Parameterized Softmax distribution.
    A categorical distribution that is determined both by the type of an action and its parameters
    using multiple softmax distributions for each action_type and for each parameter.


    Args:
        logits (ndarray or chainer.Variable): Logits for action_type and parameters softmax
            distributions.
        parametric_segments(tuple[tuple[int, optional]]): a tuple that contains tuple for each action type,
            that contains in its turn the number of
            options to choose from for each parameter
        beta (float): inverse of the temperature parameter of softmax
            distribution
        min_prob (float): minimum probability across all labels
    """

    def __init__(self, logits, parametric_segments, parametric_segments_sizes, beta=1.0, min_prob=0.0):
        self.logits = logits
        self.beta = beta
        self.min_prob = min_prob
        self.parametric_segments = parametric_segments
        self.segments_sizes = parametric_segments_sizes
        self.n = logits.shape[1]
        assert self.min_prob * self.n <= 1.0

    def k_highest_probablities(self, k):
        indices = np.argpartition(self.all_prob.array, -k,  axis=1)[:, -k:]
        k_highest_probs = np.array([self.all_prob.data[i, indices[i, :]] for i in range(len(self.all_prob))])
        # sort probabilities in descending order
        sorted_k_highest_probs = -np.sort(-k_highest_probs, axis=1)

        return sorted_k_highest_probs.astype(np.float32)

    @property
    def params(self):
        return (self.logits,)

    @cached_property
    def all_prob(self):
        with chainer.force_backprop_mode():
            if self.min_prob > 0:
                assert False
                return (F.softmax(self.beta * self.logits)
                        * (1 - self.min_prob * self.n)) + self.min_prob
            else:
                # consider to use something like https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/distributions.html#MultiCategoricalProbabilityDistribution
                # and https://stable-baselines.readthedocs.io/en/master/common/distributions.html
                return self.get_all_prob_or_log_prob(is_log=False)

    @cached_property
    def all_log_prob(self):
        with chainer.force_backprop_mode():
            if self.min_prob > 0:
                assert False
                return F.log(self.all_prob)
            else:
                return self.get_all_prob_or_log_prob(is_log=True)

    def get_all_prob_or_log_prob(self, is_log=False):
        segments = self.parametric_segments
        num_of_action_types = len(segments)
        action_types_logits = self.beta * self.logits[:, :num_of_action_types]
        if is_log:
            action_types_probs = F.log_softmax(action_types_logits)
        else:
            action_types_probs = F.softmax(action_types_logits)
        # action_types_probs = F.softmax(action_types_logits)
        # if is_log:
        #     print("LOG")
        # print(action_types_probs)
        # action_types_probs = action_types_probs.data * np.power(self.segments_sizes, 1/4)
        # action_types_probs = action_types_probs / np.expand_dims(np.sum(action_types_probs, axis=1), axis=1)
        # action_types_probs = chainer.Variable(action_types_probs.astype(np.float32))
        # if is_log:
        #     action_types_probs = F.log(action_types_probs)
        # print(action_types_probs)

        result = []
        logits_offset = num_of_action_types
        for i in range(num_of_action_types):
            action_type_prob = action_types_probs[:, i:i + 1]
            if not segments[i]:  # if no parameters for this action type
                result.append(action_type_prob)
            else:
                segments_factor = 1
                for sub_seg_size in segments[i]:
                    segments_factor *= sub_seg_size
                    if is_log:
                        sub_seg_probs = F.log_softmax(self.beta * self.logits[:, logits_offset:logits_offset + sub_seg_size])
                    else:
                        sub_seg_probs = F.softmax(self.beta * self.logits[:, logits_offset:logits_offset + sub_seg_size])
                    if is_log:
                        action_type_prob = F.repeat(action_type_prob, sub_seg_size, axis=1) + F.tile(sub_seg_probs,
                                                                                                     segments_factor // sub_seg_size)
                    else:
                        action_type_prob = F.repeat(action_type_prob, sub_seg_size, axis=1) * F.tile(sub_seg_probs,
                                                                                                     segments_factor // sub_seg_size)
                logits_offset += sub_seg_size
                result.append(action_type_prob)

        res = F.concat(tuple(result))
        return res

    def copy(self):
        return ParamSoftmaxDistribution(distribution._unwrap_variable(self.logits).copy(),
                                        parametric_segments=self.parametric_segments,
                                        parametric_segments_sizes=self.segments_sizes,
                                        beta=self.beta, min_prob=self.min_prob)

    def __repr__(self):
        return 'ParamSoftmaxDistribution(beta={}, min_prob={}) logits:{} parametric_segments:{} probs:{} entropy:{}'.format(  # NOQA
            self.beta, self.min_prob, self.logits.array, self.parametric_segments,
            self.all_prob.array, self.entropy.array)

    def __getitem__(self, i):
        return ParamSoftmaxDistribution(self.logits[i],
                                             parametric_segments=self.parametric_segments,
                                             parametric_segments_sizes=self.segments_sizes,
                                             beta=self.beta, min_prob=self.min_prob)


class ParamSoftmaxPolicy(chainer.Chain, Policy):
    """Parameterized Softmax policy that uses Boltzmann distributions.

    Args:
        model (chainer.Link):
            Link that is callable and outputs action values.
        parametric_segments(tuple[tuple[int, optional]]): a tuple that contains tuple for each action type,
            that contains in its turn the number of
        beta (float):
            Parameter of Boltzmann distributions.
    """

    def __init__(self, model, parametric_segments, parametric_segments_sizes, beta=1.0, min_prob=0.0):
        self.beta = beta
        self.min_prob = min_prob
        self.parametric_segments = parametric_segments
        self.parametric_segments_sizes = parametric_segments_sizes
        super().__init__(model=model)

    def __call__(self, x):
        h = self.model(x)
        return ParamSoftmaxDistribution(logits=h, parametric_segments=self.parametric_segments,
                                        parametric_segments_sizes=self.parametric_segments_sizes,
                                        beta=self.beta, min_prob=self.min_prob)


class FCParamSoftmaxPolicy(ParamSoftmaxPolicy):
    """Parameterized Softmax policy that consists of FC layers and rectifiers"""

    def __init__(self, n_input_channels, n_discrete_entries, parametric_segments, parametric_segments_sizes,
                 n_hidden_layers=0, n_hidden_channels=None,
                 beta=1.0, nonlinearity=F.relu,
                 last_wscale=1.0,
                 min_prob=0.0):
        self.n_input_channels = n_input_channels
        self.n_discrete_entries = n_discrete_entries
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.beta = beta

        super().__init__(
            model=MLP(n_input_channels,
                      n_discrete_entries,
                      (n_hidden_channels,) * n_hidden_layers,
                      nonlinearity=nonlinearity,
                      last_wscale=last_wscale),
            parametric_segments=parametric_segments,
            parametric_segments_sizes=parametric_segments_sizes,
            beta=self.beta,
            min_prob=min_prob)


class TestParameterizedSoftmaxDistribution(unittest.TestCase):

    def test_parameterized_softmax_distr(self):
        import numpy as np

        """test output dimension"""
        segs = (tuple(), (12, 3, 11), (12,))
        x = np.arange(39, dtype=np.float32)
        x2 = 2 * x
        logits = F.stack((x, x2))
        beta = 1.0

        parameterized_distr = ParamSoftmaxDistribution(logits, segs, beta)

        p = parameterized_distr.all_prob
        log_p = parameterized_distr.all_log_prob

        # sanity check
        assert p.shape[1] == 1 + 12 * 3 * 11 + 12
        assert log_p.shape[1] == 1 + 12 * 3 * 11 + 12

        """test probabilities and log probabilities calculated"""
        x = np.arange(15, dtype=np.float32)
        x2 = 2 * x
        logits = F.stack((x, x2))

        num_of_cols = 6

        def probs():
            action_type = F.softmax(beta * logits[:, :3])
            back_prob = action_type[:, 0:1]
            filter_prob = action_type[:, 1:2]
            group_prob = action_type[:, 2:3]
            filter_col_prob = F.softmax(beta * logits[:, 3:3 + num_of_cols])
            filter_col_prob = F.broadcast_to(filter_prob, filter_col_prob.shape) * filter_col_prob
            group_col_prob = F.softmax(beta * logits[:, 3 + num_of_cols:3 + num_of_cols + num_of_cols])
            group_col_prob = F.broadcast_to(group_prob, group_col_prob.shape) * group_col_prob

            res = F.concat((back_prob, filter_col_prob, group_col_prob))
            # sanity check, sum(result, axis=1) == 1
            assert np.all(F.sum(res, axis=1).data == 1.0)

            return res

        def log_probs():
            action_type = F.log_softmax(beta * logits[:, :3])
            back_prob = action_type[:, 0:1]
            filter_prob = action_type[:, 1:2]
            group_prob = action_type[:, 2:3]
            filter_col_prob = F.log_softmax(beta * logits[:, 3:3 + num_of_cols])
            filter_col_prob = F.broadcast_to(filter_prob, filter_col_prob.shape) + filter_col_prob
            group_col_prob = F.log_softmax(beta * logits[:, 3 + num_of_cols:3 + num_of_cols + num_of_cols])
            group_col_prob = F.broadcast_to(group_prob, group_col_prob.shape) + group_col_prob

            res = F.concat((back_prob, filter_col_prob, group_col_prob))
            return res


        segs = (tuple(), (num_of_cols,), (num_of_cols,))
        parameterized_distr = ParamSoftmaxDistribution(logits, segs, beta)

        p = probs()
        p2 = parameterized_distr.all_prob

        # sanity check 2: probs() == generic_probs()
        assert np.allclose(p.data, p2.data)

        log_p = log_probs()
        log_p2 = parameterized_distr.all_log_prob

        # sanity check 3: probs() == generic_probs()
        assert np.allclose(log_p.data, log_p2.data)

        # sanity check 4: log(p)==log_p
        assert np.allclose(F.log(p).data, log_p.data)


if __name__ == '__main__':
    unittest.main()
