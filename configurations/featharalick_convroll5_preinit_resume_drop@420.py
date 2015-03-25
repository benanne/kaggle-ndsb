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

features = [
    # "hu",
    # "tutorial",
    "haralick",
    # "aaronmoments",
    # "lbp",
    # "pftas",
    # "zernike_moments",
    # "image_size",
]

batch_size = 128
chunk_size = 32768
num_chunks_train = 240

momentum = 0.9
learning_rate_schedule = {
    0: 0.001,
    100: 0.0001,
    200: 0.00001,
}

validate_every = 40
save_every = 40

sdir = "/mnt/storage/users/avdnoord/git/kaggle-plankton/predictions/"
train_pred_file = sdir+"train--convroll5_preinit_resume_drop@420--convroll5_preinit_resume_drop@420-schaap-20150225-131159--avg-probs.npy"
valid_pred_file = sdir+"valid--convroll5_preinit_resume_drop@420--convroll5_preinit_resume_drop@420-schaap-20150225-131159--avg-probs.npy"
test_pred_file =  sdir+"test--convroll5_preinit_resume_drop@420--convroll5_preinit_resume_drop@420-schaap-20150225-131159--avg-probs.npy"

data_loader = load.PredictionsWithFeaturesDataLoader(
    features = features,
    train_pred_file=train_pred_file, 
    valid_pred_file=valid_pred_file, 
    test_pred_file=test_pred_file,
    num_chunks_train=num_chunks_train, 
    chunk_size=chunk_size)

create_train_gen = lambda: data_loader.create_random_gen()
create_eval_train_gen = lambda: data_loader.create_fixed_gen("train")
create_eval_valid_gen = lambda: data_loader.create_fixed_gen("valid")
create_eval_test_gen = lambda: data_loader.create_fixed_gen("test")

def build_model():
    l0 = nn.layers.InputLayer((batch_size, data.num_classes))

    l0_size = nn.layers.InputLayer((batch_size, 52))
    l1_size = nn.layers.DenseLayer(l0_size, num_units=80, W=nn_plankton.Orthogonal('relu'), b=nn.init.Constant(0.1))
    l2_size = nn.layers.DenseLayer(l1_size, num_units=80, W=nn_plankton.Orthogonal('relu'), b=nn.init.Constant(0.1))
    l3_size = nn.layers.DenseLayer(l2_size, num_units=data.num_classes, W=nn_plankton.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=None)

    l1 = nn_plankton.NonlinLayer(l0, T.log)
    ltot = nn.layers.ElemwiseSumLayer([l1, l3_size])

    # norm_by_sum = lambda x: x / x.sum(1).dimshuffle(0, "x")
    lout = nn_plankton.NonlinLayer(ltot, nonlinearity=T.nnet.softmax)

    return [l0, l0_size], lout



def build_objective(l_ins, l_out):
    reg_param = 0.0002
    alpha = 0. # 0 -> L2  1-> L1
    print "regu", reg_param, alpha
    # lambda_reg = 0.005
    params = nn.layers.get_all_non_bias_params(l_out)
    # reg_term = sum(T.sum(p**2) for p in params)
    L2 = sum(T.sum(p**2) for p in params)
    L1 = sum(T.sum(T.abs_(p)) for p in params)

    def loss(y, t):
        return nn_plankton.log_loss(y, t) + reg_param*(alpha * L1 + (1-alpha) * L2)

    return nn.objectives.Objective(l_out, loss_function=loss)