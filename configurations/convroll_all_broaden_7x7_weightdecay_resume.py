import numpy as np

import theano 
import theano.tensor as T

import lasagne as nn

import data
import load
import nn_plankton
import dihedral
import dihedral_fast
import tmp_dnn
import tta


resume_path = "metadata/convroll_all_broaden_7x7_weightdecay-paard-20150219-135707.pkl"


patch_size = (92, 92)
augmentation_params = {
    'zoom_range': (1 / 1.6, 1.6),
    'rotation_range': (0, 360),
    'shear_range': (-20, 20),
    'translation_range': (-10, 10),
    'do_flip': True,
    'allow_stretch': 1.3,
}

batch_size = 128 // 4
chunk_size = 32768 // 4
num_chunks_train = 840

momentum = 0.9
learning_rate_schedule = {
    0: 0.0015,
    700: 0.00015,
    800: 0.000015,
}

validate_every = 20
save_every = 20


def estimate_scale(img):
    return np.maximum(img.shape[0], img.shape[1]) / 85.0
    

# augmentation_transforms_test = []
# for flip in [True, False]:
#     for zoom in [1/1.3, 1/1.2, 1/1.1, 1.0, 1.1, 1.2, 1.3]:
#         for rot in np.linspace(0.0, 360.0, 5, endpoint=False):
#             tf = data.build_augmentation_transform(zoom=(zoom, zoom), rotation=rot, flip=flip)
#             augmentation_transforms_test.append(tf)
augmentation_transforms_test = tta.build_quasirandom_transforms(70, **{
    'zoom_range': (1 / 1.4, 1.4),
    'rotation_range': (0, 360),
    'shear_range': (-10, 10),
    'translation_range': (-8, 8),
    'do_flip': True,
    'allow_stretch': 1.2,
})



data_loader = load.ZmuvRescaledDataLoader(estimate_scale=estimate_scale, num_chunks_train=num_chunks_train,
    patch_size=patch_size, chunk_size=chunk_size, augmentation_params=augmentation_params,
    augmentation_transforms_test=augmentation_transforms_test)


# Conv2DLayer = nn.layers.cuda_convnet.Conv2DCCLayer
# MaxPool2DLayer = nn.layers.cuda_convnet.MaxPool2DCCLayer

Conv2DLayer = tmp_dnn.Conv2DDNNLayer
MaxPool2DLayer = tmp_dnn.MaxPool2DDNNLayer


def conv(incoming, **kwargs):
    return Conv2DLayer(incoming, border_mode="same",
        W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1),
        nonlinearity=nn_plankton.leaky_relu, untie_biases=True,
        **kwargs)


convroll = dihedral_fast.CyclicConvRollLayer


def pool(incoming, **kwargs):
    return MaxPool2DLayer(incoming, ds=(3, 3), strides=(2, 2), **kwargs)


def build_model():
    l0 = nn.layers.InputLayer((batch_size, 1, patch_size[0], patch_size[1]))
    l0c = dihedral.CyclicSliceLayer(l0)

    l1 = convroll(conv(l0c, num_filters=16, filter_size=(7, 7), strides=(2, 2)))

    l2 = convroll(conv(l1, num_filters=32, filter_size=(7, 7), strides=(2, 2)))
    
    l3a = convroll(conv(l2, num_filters=32, filter_size=(3, 3)))
    l3b = convroll(conv(l3a, num_filters=32, filter_size=(3, 3)))
    l3c = convroll(conv(l3b, num_filters=32, filter_size=(3, 3)))
    l3d = conv(l3c, num_filters=64, filter_size=(3, 3))
    l3 = convroll(pool(l3d))

    l4a = convroll(conv(l3, num_filters=64, filter_size=(3, 3)))
    l4b = convroll(conv(l4a, num_filters=64, filter_size=(3, 3)))
    l4c = convroll(conv(l4b, num_filters=64, filter_size=(3, 3)))
    l4d = conv(l4c, num_filters=128, filter_size=(3, 3))
    l4 = convroll(pool(l4d))

    l5a = convroll(conv(l4, num_filters=64, filter_size=(3, 3)))
    l5b = convroll(conv(l5a, num_filters=64, filter_size=(3, 3)))
    l5c = convroll(conv(l5b, num_filters=64, filter_size=(3, 3)))
    l5d = conv(l5c, num_filters=128, filter_size=(3, 3))
    l5 = convroll(pool(l5d))
    l5f = nn.layers.flatten(l5)

    l6 = nn.layers.DenseLayer(nn.layers.dropout(l5f, p=0.5), num_units=256, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l6r = dihedral_fast.CyclicRollLayer(l6)

    l7 = nn.layers.DenseLayer(nn.layers.dropout(l6r, p=0.5), num_units=256, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l7m = dihedral.CyclicPoolLayer(l7, pool_function=nn_plankton.rms)

    l8 = nn.layers.DenseLayer(nn.layers.dropout(l7m, p=0.5), num_units=data.num_classes, nonlinearity=T.nnet.softmax, W=nn_plankton.Orthogonal(1.0))

    return [l0], l8


def build_objective(l_ins, l_out):
    lambda_reg = 0.0005
    params = nn.layers.get_all_non_bias_params(l_out)
    reg_term = sum(T.sum(p**2) for p in params)

    def loss(y, t):
        return nn_plankton.log_loss(y, t) + lambda_reg * reg_term

    return nn.objectives.Objective(l_out, loss_function=loss)