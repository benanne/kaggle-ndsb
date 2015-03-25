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


validation_split_path = "splits/bagging_split_19.pkl"


patch_sizes = [(95, 95), (47, 47)]
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
    0: 0.003,
    700: 0.0003,
    800: 0.00003,
}

validate_every = 20
save_every = 20


def estimate_scale(img):
    return np.maximum(img.shape[0], img.shape[1]) / 85.0

scale_factors = [estimate_scale, 5.0]  # combine size-based rescaling + fixed rescaling
    

augmentation_transforms_test = tta.build_quasirandom_transforms(70, **{
    'zoom_range': (1 / 1.4, 1.4),
    'rotation_range': (0, 360),
    'shear_range': (-10, 10),
    'translation_range': (-8, 8),
    'do_flip': True,
    'allow_stretch': 1.2,
})



data_loader = load.ZmuvMultiscaleDataLoader(scale_factors=scale_factors, num_chunks_train=num_chunks_train,
    patch_sizes=patch_sizes, chunk_size=chunk_size, augmentation_params=augmentation_params,
    augmentation_transforms_test=augmentation_transforms_test, validation_split_path=validation_split_path)


# Conv2DLayer = nn.layers.cuda_convnet.Conv2DCCLayer
# MaxPool2DLayer = nn.layers.cuda_convnet.MaxPool2DCCLayer

Conv2DLayer = tmp_dnn.Conv2DDNNLayer
MaxPool2DLayer = tmp_dnn.MaxPool2DDNNLayer


def build_model():
    # variable scale part
    l0_variable = nn.layers.InputLayer((batch_size, 1, patch_sizes[0][0], patch_sizes[0][1]))
    l0c = dihedral.CyclicSliceLayer(l0_variable)

    l1a = Conv2DLayer(l0c, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l1b = Conv2DLayer(l1a, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l1 = MaxPool2DLayer(l1b, ds=(3, 3), strides=(2, 2))
    l1r = dihedral_fast.CyclicConvRollLayer(l1)

    l2a = Conv2DLayer(l1r, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l2b = Conv2DLayer(l2a, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l2 = MaxPool2DLayer(l2b, ds=(3, 3), strides=(2, 2))
    l2r = dihedral_fast.CyclicConvRollLayer(l2)

    l3a = Conv2DLayer(l2r, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3b = Conv2DLayer(l3a, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3c = Conv2DLayer(l3b, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3 = MaxPool2DLayer(l3c, ds=(3, 3), strides=(2, 2))
    l3r = dihedral_fast.CyclicConvRollLayer(l3)

    l4a = Conv2DLayer(l3r, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l4b = Conv2DLayer(l4a, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l4c = Conv2DLayer(l4b, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l4 = MaxPool2DLayer(l4c, ds=(3, 3), strides=(2, 2))
    l4r = dihedral_fast.CyclicConvRollLayer(l4)
    l4f = nn.layers.flatten(l4r)

    l5 = nn.layers.DenseLayer(nn.layers.dropout(l4f, p=0.5), num_units=1024, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l5fp = nn.layers.FeaturePoolLayer(l5, ds=2)
    l5m = dihedral.CyclicPoolLayer(l5fp, pool_function=nn_plankton.rms)

    l6 = nn.layers.DenseLayer(nn.layers.dropout(l5m, p=0.5), num_units=1024, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l6fp = nn.layers.FeaturePoolLayer(l6, ds=2)

    l_variable = l6fp


    # fixed scale part
    l0_fixed = nn.layers.InputLayer((batch_size, 1, patch_sizes[1][0], patch_sizes[1][1]))
    l0c = dihedral.CyclicSliceLayer(l0_fixed)

    l1a = Conv2DLayer(l0c, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l1b = Conv2DLayer(l1a, num_filters=8, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l1 = MaxPool2DLayer(l1b, ds=(3, 3), strides=(2, 2))
    l1r = dihedral_fast.CyclicConvRollLayer(l1)

    l2a = Conv2DLayer(l1r, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l2b = Conv2DLayer(l2a, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l2 = MaxPool2DLayer(l2b, ds=(3, 3), strides=(2, 2))
    l2r = dihedral_fast.CyclicConvRollLayer(l2)

    l3a = Conv2DLayer(l2r, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3b = Conv2DLayer(l3a, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3c = Conv2DLayer(l3b, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3 = MaxPool2DLayer(l3c, ds=(3, 3), strides=(2, 2))
    l3r = dihedral_fast.CyclicConvRollLayer(l3)
    l3f = nn.layers.flatten(l3r)

    l4 = nn.layers.DenseLayer(nn.layers.dropout(l3f, p=0.5), num_units=512, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l4fp = nn.layers.FeaturePoolLayer(l4, ds=2)
    l4m = dihedral.CyclicPoolLayer(l4fp, pool_function=nn_plankton.rms)

    l5 = nn.layers.DenseLayer(nn.layers.dropout(l4m, p=0.5), num_units=512, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l5fp = nn.layers.FeaturePoolLayer(l5, ds=2)

    l_fixed = l5fp


    # merge the parts
    l_merged = nn.layers.concat([l_variable, l_fixed])

    l7 = nn.layers.DenseLayer(nn.layers.dropout(l_merged, p=0.5), num_units=data.num_classes, nonlinearity=T.nnet.softmax, W=nn_plankton.Orthogonal(1.0))

    return [l0_variable, l0_fixed], l7


def build_objective(l_ins, l_out):
    lambda_reg = 0.0005
    params = nn.layers.get_all_non_bias_params(l_out)
    reg_term = sum(T.sum(p**2) for p in params)

    def loss(y, t):
        return nn_plankton.log_loss(y, t) + lambda_reg * reg_term

    return nn.objectives.Objective(l_out, loss_function=loss)
