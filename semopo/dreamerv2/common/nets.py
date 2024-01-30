import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

from .. import common


class EnsembleRSSM(common.Module):

    def __init__(
            self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
            act='elu', norm='none', std_act='softplus', min_std=0.1):
        super().__init__()
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._cell = GRUCell(self._deter, norm=True)
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            state = dict(
                logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype))
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype))
        return state

    @tf.function
    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (swap(action), swap(embed), swap(is_first)), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(self.img_step, action, state)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = self._cast(state['stoch'])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return tf.concat([stoch, state['deter']], -1)

    def get_dist(self, state, ensemble=False):
        if ensemble:
            state = self._suff_stats_ensemble(state['deter'])
        if self._discrete:
            logit = state['logit']
            logit = tf.cast(logit, tf.float32)
            dist = tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            mean = tf.cast(mean, tf.float32)
            std = tf.cast(std, tf.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # if is_first.any():
        prev_state, prev_action = tf.nest.map_structure(
            lambda x: tf.einsum(
                'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
            (prev_state, prev_action))
        prior = self.img_step(prev_state, prev_action, sample)
        x = tf.concat([prior['deter'], embed], -1)
        x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
        x = self.get('obs_out_norm', NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer('obs_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = self._cast(prev_state['stoch'])
        prev_action = self._cast(prev_action)
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.get('img_in', tfkl.Dense, self._hidden)(x)
        x = self.get('img_in_norm', NormLayer, self._norm)(x)
        x = self._act(x)
        deter = prev_state['deter']
        deter, _ = self._cell(x, [deter])
        stats = self._suff_stats_ensemble(deter)
        index = tf.random.uniform((), 0, self._ensemble, tf.int32)
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def _suff_stats_ensemble(self, inp):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
            x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
            x = self._act(x)
            stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
        stats = {
            k: tf.stack([x[k] for x in stats], 0)
            for k, v in stats[0].items()}
        stats = {
            k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
            for k, v in stats.items()}
        return stats

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {'logit': logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            std = {
                'softplus': lambda: tf.nn.softplus(std),
                'sigmoid': lambda: tf.nn.sigmoid(std),
                'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        kld = tfd.kl_divergence
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = tf.maximum(value, free).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = tf.maximum(value_lhs.mean(), free)
                loss_rhs = tf.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = tf.maximum(value_lhs, free).mean()
                loss_rhs = tf.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value

class StochasticRSSM(common.Module):

    def __init__(
        self,
        ensemble=5,
        stoch=30,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act=tf.nn.elu,
        norm='none',
        mean_act="none",
        std_act="softplus",
        temp_post=True,
        min_std=0.1,
        cell="keras",
    ):
        super().__init__()
        self._ensemble = ensemble
        self._stoch = stoch
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        self._act = act
        self._norm = norm
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._embed = None

        if cell == "sgru":
            self._cell = StochasticGRUCell(self._stoch, return_output=False)
            self._is_cell_stochastic = True
        elif cell == 'sgru_layernorm':
            self._cell = StochasticGRUCell(self._stoch, norm=True, return_output=False)
            self._is_cell_stochastic = True
        else:
            raise NotImplementedError(cell)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        # print (' in inital ', dtype)
        if self._discrete:
            state = dict(
                logit=tf.ones([batch_size, self._stoch, self._discrete], dtype)
                / self._discrete,
                stoch=tf.ones([batch_size, self._stoch, self._discrete], dtype)
                / self._discrete,
                cand=self._cell.get_initial_state(None, batch_size, dtype),
            )
            # state['stoch'][:,:,0] = 1.0
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
                cand=self._cell.get_initial_state(None, batch_size, dtype)
            )

        if self._is_cell_stochastic:
            state["u_sample"] = tf.zeros([batch_size, self._stoch], dtype)
            state["u_logit"] = tf.zeros([batch_size, self._stoch], dtype)

        return state

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return stoch

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            logit = tf.cast(logit, tf.float32)
            dist = tools.OneHotDist(probs=logit)
            if dtype != tf.float32:
                dist = tools.DtypeDist(dist, dtype or state["logit"].dtype)
        else:
            mean, std = state["mean"], state["std"]
            if dtype:
                mean = tf.cast(mean, dtype)
                std = tf.cast(std, dtype)
            dist = tfd.Normal(mean, std)
        return dist

    def sample_dist(self, stats, sample=True):
        if sample:
            return self.get_dist(stats).sample()
        else:
            return self.get_dist(stats).mode()

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, sample=True):
        if not self._embed:
            self._embed = embed.shape[-1]
        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                x = tf.concat([prior["cand"], embed], -1)
            else:
                x = embed
            for i in range(self._layers_output):
                x = self.get(f"obi{i}", tfkl.Dense, self._hidden, self._act)(x)

            stats = self._suff_stats_layer("obs", x)

            if self._is_cell_stochastic:
                if self._discrete:
                    stats["logit"] = (
                        tf.expand_dims(prior["u_logit"], axis=-1)
                        * tf.nn.softmax(stats["logit"], axis=-1)
                        + (1.0 - tf.expand_dims(prior["u_logit"], axis=-1))
                        * prev_state["stoch"]
                    )
                    output = self.sample_dist(stats, sample)
                else:
                    stoch = self.sample_dist(stats, sample)
                    output = (
                        prior["u_sample"] * stoch
                        + (1.0 - prior["u_sample"]) * prev_state["stoch"]
                    )

                post = {"stoch": output, "cand": prior["cand"], **stats}
                post["u_sample"] = prior["u_sample"]
                post["u_logit"] = prior["u_logit"]
            else:
                stoch = self.sample_dist(stats, sample)
                post = {"stoch": stoch, "cand": prior["cand"], **stats}

        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        if self._shared:
            if embed is None:
                shape = prev_action.shape[:-1] + [self._embed]
                embed = tf.zeros(shape, prev_action.dtype)
            x = tf.concat([prev_stoch, prev_action, embed], -1)
        else:
            x = tf.concat([prev_stoch, prev_action], -1)
        for i in range(self._layers_input):
            x = self.get(f"ini{i}", tfkl.Dense, self._hidden, self._act)(x)

        for _ in range(self._rec_depth):
            if self._is_cell_stochastic:
                deter = [prev_stoch, prev_state["u_sample"], prev_state["u_logit"]]
            else:
                deter = [prev_state["stoch"]]
            x, deter = self._cell(x, deter)

        for i in range(self._layers_output):
            x = self.get(f"imo{i}", tfkl.Dense, self._hidden, self._act)(x)

        stats = self._suff_stats_layer("ims", x)

        if self._is_cell_stochastic:
            if self._discrete:
                stats["logit"] = (
                    tf.expand_dims(deter[2], axis=-1)
                    * tf.nn.softmax(stats["logit"], axis=-1)
                    + (1.0 - tf.expand_dims(deter[2], axis=-1)) * prev_state["stoch"]
                )
                output = self.sample_dist(stats, sample)
            else:
                stoch = self.sample_dist(stats, sample)
                output = deter[1] * stoch + (1.0 - deter[1]) * prev_stoch

            prior = {"stoch": output, "cand": deter[0], **stats}
            prior["u_sample"] = deter[1]
            prior["u_logit"] = deter[2]
        else:
            stoch = self.sample_dist(stats, sample)
            prior = {"stoch": stoch, "cand": deter[0], **stats}

        return prior

    def _suff_stats_ensemble(self, inp):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
            x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
            x = self._act(x)
            stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
        stats = {
            k: tf.stack([x[k] for x in stats], 0)
            for k, v in stats[0].items()}
        stats = {
            k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
            for k, v in stats.items()}
        return stats

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * tf.math.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: tf.nn.softplus(std),
                "abs": lambda: tf.math.abs(std + 1),
                "sigmoid": lambda: tf.nn.sigmoid(std),
                "sigmoid2": lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    @tf.function
    def observe(self, embed, action, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        embed, action = swap(embed), swap(action)
        post, prior = static_scan(
                lambda prev, inputs: self.obs_step(prev[0], *inputs),
                (action, embed), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = static_scan(self.img_step, action, state)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def sparsity_loss(self, post, prior_prob, free, scale):
        kld = tfd.kl_divergence
        u_post = tfd.Independent(
                BernoulliDist(probs=tf.cast(post["u_logit"], tf.float32)), 1
        )
        u_prior = tfd.Independent(
                BernoulliDist(probs=tf.ones(post["u_logit"].shape) * prior_prob), 1
        )
        loss = kld(u_post, u_prior)
        loss = tf.maximum(tf.reduce_mean(loss), free)
        return loss * scale

class Encoder(common.Module):

    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        self.shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
        print('Encoder CNN inputs:', list(self.cnn_keys))
        print('Encoder MLP inputs:', list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    @tf.function
    def __call__(self, data):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[:-len(shape)]
        data = {
            k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
            for k, v in data.items()}
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = tf.concat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])

    def _cnn(self, data):
        x = tf.concat(list(data.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** i * self._cnn_depth
            x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
            x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        return x.reshape(tuple(x.shape[:-3]) + (-1,))

    def _mlp(self, data):
        x = tf.concat(list(data.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f'dense{i}', tfkl.Dense, width)(x)
            x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        return x


class Decoder(common.Module):

    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
        print('Decoder CNN outputs:', list(self.cnn_keys))
        print('Decoder MLP outputs:', list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def __call__(self, features):
        features = tf.cast(features, prec.global_policy().compute_dtype)
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    def _cnn(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}# {'image':3}
        ConvT = tfkl.Conv2DTranspose
        x = self.get('convin', tfkl.Dense, 32 * self._cnn_depth)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._cnn_depth])
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
            act, norm = self._act, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act, norm = sum(channels.values()), tf.identity, 'none'
            x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
            x = self.get(f'convnorm{i}', NormLayer, norm)(x)
            x = act(x)
        x = x.reshape(features.shape[:-1] + x.shape[1:])
        means = tf.split(x, list(channels.values()), -1) # (h, w, 3)
        dists = {
            key: tfd.Independent(tfd.Normal(mean, 1), 3)
            for (key, shape), mean in zip(channels.items(), means)}
        return dists # {'image': }

    def _mlp(self, features):
        shapes = {k: self._shapes[k] for k in self.mlp_keys}
        x = features
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f'dense{i}', tfkl.Dense, width)(x)
            x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        dists = {}
        for key, shape in shapes.items():
            dists[key] = self.get(f'dense_{key}', DistLayer, shape)(x)
        return dists

class DecoderMask(common.Module):
    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
        print('Decoder CNN outputs:', list(self.cnn_keys))
        print('Decoder MLP outputs:', list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def __call__(self, features):
        features = tf.cast(features, prec.global_policy().compute_dtype)
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    def _cnn(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys} # {'image':3}
        ConvT = tfkl.Conv2DTranspose
        x = self.get('convin', tfkl.Dense, 32 * self._cnn_depth)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._cnn_depth])
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
            act, norm = self._act, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act, norm = sum(channels.values())*2, tf.identity, 'none'
            x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
            x = self.get(f'convnorm{i}', NormLayer, norm)(x)
            x = act(x)
        x = x.reshape(features.shape[:-1] + x.shape[1:])
        mean, mask = tf.split(x, list(channels.values())+list(channels.values()), -1)
        dists = {
            'image': tfd.Independent(tfd.Normal(mean, 1), 3),
            'mask': mask}
        return dists

    def _mlp(self, features):
        shapes = {k: self._shapes[k] for k in self.mlp_keys}
        x = features
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f'dense{i}', tfkl.Dense, width)(x)
            x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        dists = {}
        for key, shape in shapes.items():
            dists[key] = self.get(f'dense_{key}', DistLayer, shape)(x)
        return dists
    
      

class DecoderMaskEnsemble(common.Module):
    """
    ensemble two convdecoder with <Normal, mask> outputs
    NOTE: remove pred1/pred2 for maximum performance.
    """

    def __init__(self, decoder1, decoder2):
        self._decoder1 = decoder1
        self._decoder2 = decoder2
        self._shape = decoder1._shapes['image']

    def __call__(self, feat1, feat2, dtype=tf.float32):
        kwargs = dict(strides=1, activation=tf.nn.sigmoid)
        out1 = self._decoder1(feat1)
        pred1, mask1 = out1['image'], out1['mask']
        out2 = self._decoder2(feat2)
        pred2, mask2 = out2['image'], out2['mask']
        mean1 = pred1.submodules[0].loc
        mean2 = pred2.submodules[0].loc
        mask_feat = tf.concat([mask1, mask2], -1)
        mask = self.get('mask1', tfkl.Conv2D, 1, 1, **kwargs)(mask_feat)
        mask_use1 = mask
        mask_use2 = 1-mask
        mean = mean1 * tf.cast(mask_use1, mean1.dtype) + \
            mean2 * tf.cast(mask_use2, mean2.dtype)
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape)), pred1, pred2, tf.cast(mask_use1, mean1.dtype)

class MLP(common.Module):

    def __init__(self, shape, layers, units, act='elu', norm='none', **out):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def __call__(self, features):
        x = tf.cast(features, prec.global_policy().compute_dtype)
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f'dense{index}', tfkl.Dense, self._units)(x)
            x = self.get(f'norm{index}', NormLayer, self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + [x.shape[-1]])
        return self.get('out', DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    @tf.function
    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

class StochasticGRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, return_output=True, **kwargs):
        super().__init__()
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
        self._return_output = return_output
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))

        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)

        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update_p = tf.nn.sigmoid(update + self._update_bias)
        update = tf.cast(
            tfd.Independent(tools.BernoulliDist(probs=update_p), 1).sample(), tf.float16
        )
        if self._return_output:
          output = update * cand + (1 - update) * state
          return output, [output, update, update_p]
        else:
          return cand, [cand, update, update_p]
  

class DistLayer(common.Module):

    def __init__(
            self, shape, dist='mse', min_std=0.1, init_std=0.0):
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def __call__(self, inputs):
        out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
        out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        out = tf.cast(out, tf.float32)
        if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
            std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
            std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
            std = tf.cast(std, tf.float32)
        if self._dist == 'mse':
            dist = tfd.Normal(out, 1.0)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == 'normal':
            dist = tfd.Normal(out, std)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == 'binary':
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == 'tanh_normal':
            mean = 5 * tf.tanh(out / 5)
            std = tf.nn.softplus(std + self._init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, common.TanhBijector())
            dist = tfd.Independent(dist, len(self._shape))
            return common.SampleDist(dist)
        if self._dist == 'trunc_normal':
            std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
            return tfd.Independent(dist, 1)
        if self._dist == 'onehot':
            return common.OneHotDist(out)
        raise NotImplementedError(self._dist)


class NormLayer(common.Module):

    def __init__(self, name):
        if name == 'none':
            self._layer = None
        elif name == 'layer':
            self._layer = tfkl.LayerNormalization()
        else:
            raise NotImplementedError(name)

    def __call__(self, features):
        if not self._layer:
            return features
        return self._layer(features)


def get_act(name):
    if name == 'none':
        return tf.identity
    if name == 'mish':
        return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
    elif hasattr(tf.nn, name):
        return getattr(tf.nn, name)
    elif hasattr(tf, name):
        return getattr(tf, name)
    else:
        raise NotImplementedError(name)
