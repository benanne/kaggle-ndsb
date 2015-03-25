import numpy as np
import theano.tensor as T

import lasagne as nn


# some helper functions that rotate arrays on the trailing axes.
# these should work for both Theano expressions and numpy arrays.

def array_tf_0(arr):
    return arr

def array_tf_90(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)

def array_tf_180(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]

def array_tf_270(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_0f(arr): # horizontal flip
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)]

def array_tf_90f(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None)]
    # slicing does nothing here, technically I could get rid of it.
    return arr[tuple(slices)].transpose(axes_order)

def array_tf_180f(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)]

def array_tf_270f(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)


# c01b versions of the helper functions

def array_tf_0_c01b(arr):
    return arr

def array_tf_90_c01b(arr):
    axes_order = [0, 2, 1, 3]
    slices = [slice(None), slice(None), slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)

def array_tf_180_c01b(arr):
    slices = [slice(None), slice(None, None, -1), slice(None, None, -1), slice(None)]
    return arr[tuple(slices)]

def array_tf_270_c01b(arr):
    axes_order = [0, 2, 1, 3]
    slices = [slice(None), slice(None, None, -1), slice(None), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_0f_c01b(arr): # horizontal flip
    slices = [slice(None), slice(None), slice(None, None, -1), slice(None)]
    return arr[tuple(slices)]

def array_tf_90f_c01b(arr):
    axes_order = [0, 2, 1, 3]
    slices = [slice(None), slice(None), slice(None), slice(None)]
    # slicing does nothing here, technically I could get rid of it.
    return arr[tuple(slices)].transpose(axes_order)

def array_tf_180f_c01b(arr):
    slices = [slice(None), slice(None, None, -1), slice(None), slice(None)]
    return arr[tuple(slices)]

def array_tf_270f_c01b(arr):
    axes_order = [0, 2, 1, 3]
    slices = [slice(None), slice(None, None, -1), slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)



class CyclicSliceLayer(nn.layers.Layer):
    """
    This layer stacks rotations of 0, 90, 180, and 270 degrees of the input
    along the batch dimension.

    If the input has shape (batch_size, num_channels, r, c),
    then the output will have shape (4 * batch_size, num_channels, r, c).

    Note that the stacking happens on axis 0, so a reshape to
    (4, batch_size, num_channels, r, c) will separate the slice axis.
    """
    def __init__(self, input_layer):
        super(CyclicSliceLayer, self).__init__(input_layer)

    def get_output_shape_for(self, input_shape):
        return (4 * input_shape[0],) + input_shape[1:]

    def get_output_for(self, input, *args, **kwargs):
        return nn.utils.concatenate([
                array_tf_0(input),
                array_tf_90(input),
                array_tf_180(input),
                array_tf_270(input),
            ], axis=0)


class DihedralSliceLayer(nn.layers.Layer):
    """
    This layer stacks rotations of 0, 90, 180, and 270 degrees of the input,
    as well as their horizontal flips, along the batch dimension.

    If the input has shape (batch_size, num_channels, r, c),
    then the output will have shape (8 * batch_size, num_channels, r, c).

    Note that the stacking happens on axis 0, so a reshape to
    (8, batch_size, num_channels, r, c) will separate the slice axis.
    """
    def __init__(self, input_layer):
        super(DihedralSliceLayer, self).__init__(input_layer)

    def get_output_shape_for(self, input_shape):
        return (8 * input_shape[0],) + input_shape[1:]

    def get_output_for(self, input, *args, **kwargs):
        return nn.utils.concatenate([
                array_tf_0(input),
                array_tf_90(input),
                array_tf_180(input),
                array_tf_270(input),
                array_tf_0f(input),
                array_tf_90f(input),
                array_tf_180f(input),
                array_tf_270f(input),
            ], axis=0)


class CyclicRollLayer(nn.layers.Layer):
    """
    This layer turns (n_views * batch_size, num_features) into
    (n_views * batch_size, n_views * num_features) by rolling
    and concatenating feature maps.
    """
    def __init__(self, input_layer):
        super(CyclicRollLayer, self).__init__(input_layer)
        self.compute_permutation_matrix()

    def compute_permutation_matrix(self):
        map_identity = np.arange(4)
        map_rot90 = np.array([1, 2, 3, 0])

        valid_maps = []
        current_map = map_identity
        for k in xrange(4):
            valid_maps.append(current_map)
            current_map = current_map[map_rot90]

        self.perm_matrix = np.array(valid_maps)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 4*input_shape[1])

    def get_output_for(self, input, *args, **kwargs):
        s = input.shape
        input_unfolded = input.reshape((4, s[0] // 4, s[1]))

        permuted_inputs = []
        for p in self.perm_matrix:
            input_permuted = input_unfolded[p].reshape(s)
            permuted_inputs.append(input_permuted)

        return nn.utils.concatenate(permuted_inputs, axis=1) # concatenate long the channel axis


class DihedralRollLayer(nn.layers.Layer):
    """
    This layer turns (n_views * batch_size, num_features) into
    (n_views * batch_size, n_views * num_features) by rolling
    and concatenating feature maps.
    """
    def __init__(self, input_layer):
        super(DihedralRollLayer, self).__init__(input_layer)
        self.compute_permutation_matrix()

    def compute_permutation_matrix(self):
        map_identity = np.arange(8)
        map_rot90 = np.array([1, 2, 3, 0, 5, 6, 7, 4]) # 7, 4, 5, 6]) # CORRECTED
        map_flip = np.array([4, 5, 6, 7, 0, 1, 2, 3])

        valid_maps = []
        current_map = map_identity
        for k in xrange(4):
            valid_maps.append(current_map)
            current_map = current_map[map_rot90]

        for k in xrange(4):
            valid_maps.append(current_map[map_flip])
            current_map = current_map[map_rot90]

        self.perm_matrix = np.array(valid_maps)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 8*input_shape[1])

    def get_output_for(self, input, *args, **kwargs):
        s = input.shape
        input_unfolded = input.reshape((8, s[0] // 8, s[1]))

        permuted_inputs = []
        for p in self.perm_matrix:
            input_permuted = input_unfolded[p].reshape(s)
            permuted_inputs.append(input_permuted)

        return nn.utils.concatenate(permuted_inputs, axis=1) # concatenate long the channel axis


class CyclicConvRollLayer(CyclicRollLayer):
    """
    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, n_views * num_channels, r, c) by rolling
    and concatenating feature maps.

    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.
    """
    def __init__(self, input_layer):
        super(CyclicConvRollLayer, self).__init__(input_layer)
        self.inv_tf_funcs = [array_tf_0, array_tf_270, array_tf_180, array_tf_90]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 4*input_shape[1]) + input_shape[2:]

    def get_output_for(self, input, *args, **kwargs):
        s = input.shape
        input_unfolded = input.reshape((4, s[0] // 4, s[1], s[2], s[3]))

        permuted_inputs = []
        for p, inv_tf in zip(self.perm_matrix, self.inv_tf_funcs):
            input_permuted = inv_tf(input_unfolded[p].reshape(s))
            permuted_inputs.append(input_permuted)

        return nn.utils.concatenate(permuted_inputs, axis=1) # concatenate long the channel axis


class DihedralConvRollLayer(DihedralRollLayer):
    """
    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, n_views * num_channels, r, c) by rolling
    and concatenating feature maps.

    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.
    """
    def __init__(self, input_layer):
        super(DihedralConvRollLayer, self).__init__(input_layer)
        self.inv_tf_funcs = [array_tf_0, array_tf_270, array_tf_180, array_tf_90,
                             array_tf_0f, array_tf_90f, array_tf_180f, array_tf_270f]

        raise RuntimeError("The implementation of this class is not correct.")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 8*input_shape[1]) + input_shape[2:]

    def get_output_for(self, input, *args, **kwargs):
        s = input.shape
        input_unfolded = input.reshape((8, s[0] // 8, s[1], s[2], s[3]))

        permuted_inputs = []
        for p, inv_tf in zip(self.perm_matrix, self.inv_tf_funcs):
            input_permuted = inv_tf(input_unfolded[p].reshape(s))
            permuted_inputs.append(input_permuted)

        return nn.utils.concatenate(permuted_inputs, axis=1) # concatenate along the channel axis



class CyclicConvRollLayer_c01b(CyclicConvRollLayer):
    """
    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, n_views * num_channels, r, c) by rolling
    and concatenating feature maps.

    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.
    """
    def __init__(self, input_layer):
        super(CyclicConvRollLayer, self).__init__(input_layer)
        self.inv_tf_funcs = [array_tf_0_c01b, array_tf_270_c01b, array_tf_180_c01b, array_tf_90_c01b]

    def get_output_shape_for(self, input_shape):
        return (4 * input_shape[0],) + input_shape[1:]

    def get_output_for(self, input, *args, **kwargs):
        s = input.shape
        input_unfolded = input.reshape((s[0], s[1], s[2], 4, s[3] // 4))

        permuted_inputs = []
        for p, inv_tf in zip(self.perm_matrix, self.inv_tf_funcs):
            input_permuted = inv_tf(input_unfolded[:, :, :, p, :].reshape(s))
            permuted_inputs.append(input_permuted)

        return nn.utils.concatenate(permuted_inputs, axis=0) # concatenate long the channel axis


class CyclicPoolLayer(nn.layers.Layer):
    """
    Utility layer that unfolds the viewpoints dimension and pools over it.

    Note that this only makes sense for dense representations, not for
    feature maps (because no inverse transforms are applied to align them).
    """
    def __init__(self, input_layer, pool_function=T.mean):
        super(CyclicPoolLayer, self).__init__(input_layer)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // 4, input_shape[1])

    def get_output_for(self, input, *args, **kwargs):
        unfolded_input = input.reshape((4, input.shape[0] // 4, input.shape[1]))
        return self.pool_function(unfolded_input, axis=0)


class DihedralPoolLayer(nn.layers.Layer):
    """
    Utility layer that unfolds the viewpoints dimension and pools over it.

    Note that this only makes sense for dense representations, not for
    feature maps (because no inverse transforms are applied to align them).
    """
    def __init__(self, input_layer, pool_function=T.mean):
        super(DihedralPoolLayer, self).__init__(input_layer)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // 8, input_shape[1])

    def get_output_for(self, input, *args, **kwargs):
        unfolded_input = input.reshape((8, input.shape[0] // 8, input.shape[1]))
        return self.pool_function(unfolded_input, axis=0)


class NINCyclicPoolLayer(nn.layers.Layer):
    """
    Like CyclicPoolLayer, but broadcasting along all axes beyond the first two.
    """
    def __init__(self, input_layer, pool_function=T.mean):
        super(NINCyclicPoolLayer, self).__init__(input_layer)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // 4,) + input_shape[1:]

    def get_output_for(self, input, *args, **kwargs):
        unfolded_input = input.reshape((4, self.input_shape[0] // 4) + self.input_shape[1:])
        return self.pool_function(unfolded_input, axis=0)


class FlipSliceLayer(nn.layers.Layer):
    """
    This layer stacks the input images along with their flips along the batch
    dimension.

    If the input has shape (batch_size, num_channels, r, c),
    then the output will have shape (2 * batch_size, num_channels, r, c).

    Note that the stacking happens on axis 0, so a reshape to
    (2, batch_size, num_channels, r, c) will separate the slice axis.
    """
    def __init__(self, input_layer):
        super(FlipSliceLayer, self).__init__(input_layer)

    def get_output_shape_for(self, input_shape):
        return (2 * input_shape[0],) + input_shape[1:]

    def get_output_for(self, input, *args, **kwargs):
        return nn.utils.concatenate([
                array_tf_0(input),
                array_tf_0f(input),
            ], axis=0)
