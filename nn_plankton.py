import numpy as np

import theano
import theano.tensor as T

import lasagne as nn

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

_srng = RandomStreams()


def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.sum(t * T.log(y)) / y.shape[0].astype(theano.config.floatX)
    return loss


def log_losses(y, t, eps=1e-15):
    """
    cross entropy loss per example, summed over classes
    """
    y = T.clip(y, eps, 1 - eps)
    losses = -T.sum(t * T.log(y), axis=1)
    return losses


class Orthogonal(nn.init.Initializer):
    def __init__(self, gain=1.0): # axes are the input axes.
        if gain == 'relu':
            gain = np.sqrt(2)

        self.gain = gain

    def sample(self, shape):
        if len(shape) != 2:
            raise RuntimeError("Only shapes of length 2 are supported.")

        a = np.random.normal(0.0, 1.0, shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v # pick the one with the correct shape

        # size = np.maximum(shape[0], shape[1])
        # a = np.random.normal(0.0, 1.0, (size, size))
        # q, _ = np.linalg.qr(a)

        return nn.utils.floatX(self.gain * q[:shape[0], :shape[1]])


class Conv2DOrthogonal(Orthogonal):
    """
    fan-in is considered to be the trailing 3 axes.
    """
    def sample(self, shape):
        if len(shape) != 4:
            raise RuntimeError("Only shapes of length 4 are supported.")

        fan_in = int(np.prod(shape[1:]))
        flat_shape = (shape[0], fan_in)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q_conv = q.reshape(shape)

        # size = np.maximum(shape[0], fan_in)
        # a = np.random.normal(0.0, 1.0, (size, size))
        # q, _ = np.linalg.qr(a)
        # q_conv = q[:shape[0], :fan_in].reshape(shape)

        return nn.utils.floatX(self.gain * q_conv)


class Conv2DCCOrthogonal(Conv2DOrthogonal):
    """
    cuda-convnet version (c01b arrangement)
    """
    def sample(self, shape):
        if len(shape) != 4:
            raise RuntimeError("Only shapes of length 4 are supported.")

        fan_in = int(np.prod(shape[:3]))
        flat_shape = (fan_in, shape[3])
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q_conv = q.reshape(shape)

        # size = np.maximum(shape[3], fan_in)
        # a = np.random.normal(0.0, 1.0, (size, size))
        # q, _ = np.linalg.qr(a)
        # q_conv = q[:fan_in, :shape[3]].reshape(shape)
        
        return nn.utils.floatX(self.gain * q_conv)


class NonlinLayer(nn.layers.Layer):
    """
    Layer that simply applies a nonlinearity to its input. Output shape should be the same as input shape!
    """
    def __init__(self, input_layer, nonlinearity=nn.nonlinearities.rectify):
        super(NonlinLayer, self).__init__(input_layer)
        if nonlinearity is None:
            self.nonlinearity = nn.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

    def get_output_for(self, input, *args, **kwargs):
        return self.nonlinearity(input)


class TiedDropoutLayer(nn.layers.Layer):
    """
    Dropout layer that broadcasts the mask across all axes beyond the first two.
    """
    def __init__(self, input_layer, p=0.5, rescale=True):
        super(TiedDropoutLayer, self).__init__(input_layer)
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            mask = _srng.binomial(input.shape[:2], p=retain_prob,
                                      dtype=theano.config.floatX)
            axes = [0, 1] + (['x'] * (input.ndim - 2))
            mask = mask.dimshuffle(*axes)
            return input * mask

tied_dropout = TiedDropoutLayer


class TiedDropoutLayer_c01b(TiedDropoutLayer):
    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            mask = _srng.binomial((input.shape[0], input.shape[3]), p=retain_prob,
                                      dtype=theano.config.floatX)
            mask = mask.dimshuffle([0, 'x', 'x', 1])
            return input * mask 

tied_dropout_c01b = TiedDropoutLayer_c01b






class BatchInterleaveLayer(nn.layers.MultipleInputsLayer):
    """
    Interleave multiple input batches.
    """
    def get_output_shape_for(self, input_shapes):
        s = input_shapes[0]
        assert all(shape == s for shape in input_shapes)
        return (s[0] * len(input_shapes),) + s[1:]
    
    def get_output_for(self, inputs, *args, **kwargs):
        num_inputs = len(inputs)
        out_shape = (inputs[0].shape[0] * num_inputs,) + inputs[0].shape[1:]
        out = T.zeros(out_shape)
        for k, input in enumerate(inputs):
            out = T.set_subtensor(out[k::num_inputs], input)

        return out




class ColumnFlattenLayer(nn.layers.Layer):
    """
    flatten (num_columns*batch_size, n0, n1, ..., nk)
    to (batch_size, num_columns * n1 * n2 * ... * nk)
    """
    def __init__(self, input_layer, num_columns):
        super(ColumnFlattenLayer, self).__init__(input_layer)
        self.num_columns = num_columns

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // self.num_columns, self.num_columns * np.prod(input_shape[1:]))

    def get_output_for(self, input, *args, **kwargs):
        bs = input.shape[0] // self.num_columns
        n = T.prod(input.shape[1:])
        input = input.reshape((self.num_columns, bs, n))
        return input.dimshuffle(1, 0, 2).reshape((bs, self.num_columns * n))



class GaussianDropoutLayer(nn.layers.Layer):
    """
    Implements 'Gaussian' dropout, i.e. multiplicative Gaussian noise
    instead of multiplicative Bernoulli noise.
    """
    def __init__(self, input_layer, sigma=1.0):
        super(GaussianDropoutLayer, self).__init__(input_layer)
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        if deterministic or self.sigma == 0:
            return input
        else:
            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_layer.get_output_shape()
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * _srng.normal(input_shape, avg=1.0,
                std=self.sigma, dtype=theano.config.floatX)

gaussian_dropout = GaussianDropoutLayer # shortcut


def adam(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001, epsilon=1e-8):
    """
    Adam update rule by Kingma and Ba, ICLR 2015.

    learning_rate: alpha in the paper, the step size

    beta1: exponential decay rate of the 1st moment estimate
    beta2: exponential decay rate of the 2nd moment estimate
    """
    all_grads = theano.grad(loss, all_params)
    updates = []
    
    for param_i, grad_i in zip(all_params, all_grads):
        t = theano.shared(1) # timestep, for bias correction
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX)) # 1st moment
        vparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX)) # 2nd moment

        m = beta1 * grad_i + (1 - beta1) * mparam_i # new value for 1st moment estimate
        v = beta2 * T.sqr(grad_i) + (1 - beta2) * vparam_i # new value for 2nd moment estimate
        
        m_unbiased = m / (1 - (1 - beta1) ** t.astype(theano.config.floatX))
        v_unbiased = v / (1 - (1 - beta2) ** t.astype(theano.config.floatX))
        w = param_i - learning_rate * m_unbiased / (T.sqrt(v_unbiased) + epsilon) # new parameter values

        updates.append((mparam_i, m))
        updates.append((vparam_i, v))
        updates.append((t, t + 1))
        updates.append((param_i, w))

    return updates


def adam_v2(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001, epsilon=1e-8, l_decay=1 - 1e-8):
    """
    Adam update rule by Kingma and Ba, ICLR 2015, version 2 (with momentum decay).

    learning_rate: alpha in the paper, the step size

    beta1: exponential decay rate of the 1st moment estimate
    beta2: exponential decay rate of the 2nd moment estimate
    l_decay: exponential increase rate of beta1
    """
    all_grads = theano.grad(loss, all_params)
    updates = []

    for param_i, grad_i in zip(all_params, all_grads):
        t = theano.shared(1) # timestep, for bias correction
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX)) # 1st moment
        vparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX)) # 2nd moment

        beta1_current = 1 - (1 - beta1) * l_decay ** (t.astype(theano.config.floatX) - 1)
        m = beta1_current * grad_i + (1 - beta1_current) * mparam_i # new value for 1st moment estimate
        v = beta2 * T.sqr(grad_i) + (1 - beta2) * vparam_i # new value for 2nd moment estimate
        
        m_unbiased = m / (1 - (1 - beta1) ** t.astype(theano.config.floatX))
        v_unbiased = v / (1 - (1 - beta2) ** t.astype(theano.config.floatX))
        w = param_i - learning_rate * m_unbiased / (T.sqrt(v_unbiased) + epsilon) # new parameter values

        updates.append((mparam_i, m))
        updates.append((vparam_i, v))
        updates.append((t, t + 1))
        updates.append((param_i, w))

    return updates


class BootstrapObjective(object):
    def __init__(self, input_layer, beta, mode='soft'):
        assert 0 <= beta <= 1
        assert mode in ['soft', 'hard']
        self.input_layer = input_layer
        self.target_var = T.matrix("target")
        self.beta = beta
        self.mode = mode

    def get_loss(self, input=None, target=None, *args, **kwargs):
        network_output = self.input_layer.get_output(input, *args, **kwargs)
        if target is None:
            target = self.target_var

        if self.mode == 'soft':
            aux_target = network_output 
        elif self.mode == 'hard':
            aux_target = nn.utils.one_hot(T.argmax(network_output, axis=1), m=121) # hard labels

        target = self.beta * target + (1 - self.beta) * aux_target

        return log_loss(network_output, target)


class SemiSupervisedObjective(object):
    def __init__(self, input_layer, lambda_ss=1.0):
        self.input_layer = input_layer
        self.target_var = T.matrix("target")
        self.lambda_ss = lambda_ss

    def get_loss(self, input=None, target=None, *args, **kwargs):
        network_output = self.input_layer.get_output(input, *args, **kwargs)

        if target is None:
            target = self.target_var

        labeled = target.sum(1) # because unlabeled data points have all-zero labels, this will be 1 if labeled, 0 otherwise.
        unlabeled = 1 - labeled

        obj_labeled = labeled * log_losses(network_output, target) # labeled data: cross-entropy
        obj_unlabeled = unlabeled * log_losses(network_output, network_output) # unlabeled data: entropy
        obj = obj_labeled + self.lambda_ss * obj_unlabeled
        return T.mean(obj)


class HardSemiSupervisedObjective(object):
    def __init__(self, input_layer, lambda_ss=1.0):
        self.input_layer = input_layer
        self.target_var = T.matrix("target")
        self.lambda_ss = lambda_ss

    def get_loss(self, input=None, target=None, *args, **kwargs):
        network_output = self.input_layer.get_output(input, *args, **kwargs)

        if target is None:
            target = self.target_var

        labeled = target.sum(1) # because unlabeled data points have all-zero labels, this will be 1 if labeled, 0 otherwise.
        unlabeled = 1 - labeled

        pseudo_target = T.eye(121)[T.argmax(network_output, axis=1)]
        obj_labeled = labeled * log_losses(network_output, target) # labeled data: cross-entropy
        obj_unlabeled = unlabeled * log_losses(network_output, pseudo_target) # unlabeled data: cross-entropy with pseudo targets
        obj = obj_labeled + self.lambda_ss * obj_unlabeled
        return T.mean(obj)



class AdversarialRegObjective(object):
    def __init__(self, input_layer, input_map, alpha, epsilon):
        assert 0 <= alpha <= 1
        assert epsilon >= 0
        self.input_layer = input_layer
        self.input_map = input_map # needed to get the derivative
        self.target_var = T.matrix("target")
        self.alpha = alpha
        self.epsilon = epsilon

    def get_loss(self, target=None, *args, **kwargs):
        if target is None:
            target = self.target_var

        network_output = self.input_layer.get_output(self.input_map, *args, **kwargs)
        loss = log_loss(network_output, target)
        input_grad_map = { layer: T.grad(loss, input_var) for layer, input_var in self.input_map.iteritems() }
        perturbed_input_map = { layer: input_var + self.epsilon * T.sgn(input_grad_map[layer]) for layer, input_var in self.input_map.iteritems() }
        perturbed_network_output = self.input_layer.get_output(perturbed_input_map, *args, **kwargs)        
        perturbed_loss = log_loss(perturbed_network_output, target)

        adv_loss = self.alpha * loss + (1 - self.alpha) * perturbed_loss
        return adv_loss



class SpatialDimReductionLayer(nn.layers.Layer):
    """
    Spatial dimension reduction layer. (b, c, 0, 1) -> (b, c, n)
    """
    def __init__(self, input_layer, num_units, W=nn.init.Uniform(), b=nn.init.Constant(0.), nonlinearity=nn.nonlinearities.rectify):
        super(SpatialDimReductionLayer, self).__init__(input_layer)
        if nonlinearity is None:
            self.nonlinearity = nn.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units

        output_shape = self.input_layer.get_output_shape()
        num_inputs = int(np.prod(output_shape[2:]))

        self.W = self.create_param(W, (num_inputs, num_units))
        self.b = self.create_param(b, (num_units,)) if b is not None else None

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], np.prod(input_shape[2:]))
    
    def get_output_for(self, input, *args, **kwargs):
        input_grouped = input.reshape((input.shape[0] * input.shape[1], T.prod(input.shape[2:]))) # fold b, c and fold 0, 1, ...
        activation = T.dot(input_grouped, self.W)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)

        activation = activation.reshape((input.shape[0], input.shape[1], activation.shape[1])) # unfold b, c
        return self.nonlinearity(activation)



def rms(x, axis=None, epsilon=1e-12):
    return T.sqrt(T.mean(T.sqr(x), axis=axis) + epsilon)



class CustomRescaleDropoutLayer(nn.layers.Layer):
    """
    Like dropout layer, but the train-time rescale constant can be set to a custom value.
    """
    def __init__(self, input_layer, p=0.5, rescale=1.0):
        super(CustomRescaleDropoutLayer, self).__init__(input_layer)
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            input *= np.float32(self.rescale)

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_layer.get_output_shape()
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * _srng.binomial(input_shape, p=retain_prob,
                                          dtype=theano.config.floatX)

dropout_cr = CustomRescaleDropoutLayer # shortcut


def leaky_relu(x, alpha=3.0):
    return T.maximum(x, x * (1.0 / alpha))
