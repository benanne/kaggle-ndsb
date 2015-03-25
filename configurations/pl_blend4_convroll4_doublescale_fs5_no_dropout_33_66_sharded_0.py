import numpy as np

import theano 
import theano.tensor as T

import lasagne as nn

import data
import load
import nn_plankton
import dihedral
import tmp_dnn
import tta



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
    0: 0.0015,
    700: 0.00015,
    800: 0.000015,
}

validate_every = 20
save_every = 20
keep_every = 100

test_pred_file = "predictions/weighted_blend.npy"

def estimate_scale(img):
    return np.maximum(img.shape[0], img.shape[1]) / 85.0

scale_factors = [estimate_scale, 5.0] # combine size-based rescaling + fixed rescaling
    

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



# data_loader = load.ZmuvMultiscaleDataLoader(scale_factors=scale_factors, num_chunks_train=num_chunks_train,
#     patch_sizes=patch_sizes, chunk_size=chunk_size, augmentation_params=augmentation_params,
#     augmentation_transforms_test=augmentation_transforms_test)

data_loader = load.ShardedResampledPseudolabelingZmuvMultiscaleDataLoader(shard=0,
    test_pred_file=test_pred_file, train_sample_weight=0.333,
    scale_factors=scale_factors, num_chunks_train=num_chunks_train,
    patch_sizes=patch_sizes, chunk_size=chunk_size, augmentation_params=augmentation_params,
    augmentation_transforms_test=augmentation_transforms_test)


# Conv2DLayer = nn.layers.cuda_convnet.Conv2DCCLayer
# MaxPool2DLayer = nn.layers.cuda_convnet.MaxPool2DCCLayer

Conv2DLayer = tmp_dnn.Conv2DDNNLayer
MaxPool2DLayer = tmp_dnn.MaxPool2DDNNLayer


def build_model():
    l0_variable = nn.layers.InputLayer((batch_size, 1, patch_sizes[0][0], patch_sizes[0][1]))
    l0c = dihedral.CyclicSliceLayer(l0_variable)

    l1a = Conv2DLayer(l0c, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l1b = Conv2DLayer(l1a, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l1 = MaxPool2DLayer(l1b, ds=(3, 3), strides=(2, 2))
    l1r = dihedral.CyclicConvRollLayer(l1)

    l2a = Conv2DLayer(l1r, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l2b = Conv2DLayer(l2a, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l2 = MaxPool2DLayer(l2b, ds=(3, 3), strides=(2, 2))
    l2r = dihedral.CyclicConvRollLayer(l2)

    l3a = Conv2DLayer(l2r, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l3b = Conv2DLayer(l3a, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l3c = Conv2DLayer(l3b, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l3 = MaxPool2DLayer(l3c, ds=(3, 3), strides=(2, 2))
    l3r = dihedral.CyclicConvRollLayer(l3)

    l4a = Conv2DLayer(l3r, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l4b = Conv2DLayer(l4a, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l4c = Conv2DLayer(l4b, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)    
    l4 = MaxPool2DLayer(l4c, ds=(3, 3), strides=(2, 2))
    l4r = dihedral.CyclicConvRollLayer(l4)
    l4f = nn.layers.flatten(l4r)

    l5 = nn.layers.DenseLayer(l4f, num_units=256, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l5r = dihedral.CyclicRollLayer(l5)

    l6 = nn.layers.DenseLayer(l5r, num_units=256, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l_variable = dihedral.CyclicPoolLayer(l6, pool_function=nn_plankton.rms)


    # fixed scale part
    l0_fixed = nn.layers.InputLayer((batch_size, 1, patch_sizes[1][0], patch_sizes[1][1]))
    l0c = dihedral.CyclicSliceLayer(l0_fixed)

    l1a = Conv2DLayer(l0c, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l1b = Conv2DLayer(l1a, num_filters=8, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l1 = MaxPool2DLayer(l1b, ds=(3, 3), strides=(2, 2))
    l1r = dihedral.CyclicConvRollLayer(l1)

    l2a = Conv2DLayer(l1r, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l2b = Conv2DLayer(l2a, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l2 = MaxPool2DLayer(l2b, ds=(3, 3), strides=(2, 2))
    l2r = dihedral.CyclicConvRollLayer(l2)

    l3a = Conv2DLayer(l2r, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3b = Conv2DLayer(l3a, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3c = Conv2DLayer(l3b, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3 = MaxPool2DLayer(l3c, ds=(3, 3), strides=(2, 2))
    l3r = dihedral.CyclicConvRollLayer(l3)
    l3f = nn.layers.flatten(l3r)

    l4 = nn.layers.DenseLayer(l3f, num_units=128, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1))
    l4r = dihedral.CyclicRollLayer(l4)

    l5 = nn.layers.DenseLayer(l4r, num_units=128, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1))
    l_fixed = dihedral.CyclicPoolLayer(l5, pool_function=nn_plankton.rms)    


    # merge the parts
    l_merged = nn.layers.concat([l_variable, l_fixed])

    l7 = nn.layers.DenseLayer(l_merged, num_units=data.num_classes, nonlinearity=T.nnet.softmax, W=nn_plankton.Orthogonal(1.0))

    return [l0_variable, l0_fixed], l7
