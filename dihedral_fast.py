import lasagne as nn
import dihedral_ops


class CyclicRollLayer(nn.layers.Layer):
    """
    This layer turns (n_views * batch_size, num_features) into
    (n_views * batch_size, n_views * num_features) by rolling
    and concatenating feature maps.

    fast version using a PyCUDA-based op
    """
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 4*input_shape[1])

    def get_output_for(self, input, *args, **kwargs):
        return dihedral_ops.cyclic_roll(input)


class CyclicConvRollLayer(CyclicRollLayer):
    """
    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, n_views * num_channels, r, c) by rolling
    and concatenating feature maps.

    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.

    fast version using PyCUDA-based op
    """
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 4*input_shape[1]) + input_shape[2:]

    def get_output_for(self, input, *args, **kwargs):
        return dihedral_ops.cyclic_convroll(input)
