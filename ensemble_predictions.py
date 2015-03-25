"""
Given a set of validation predictions, this script computes the optimal linear weights on the validation set.
It computes the weighted blend of test predictions, where some models are replaced by their bagged versions.
"""

import os
import numpy as np
import sys

import theano
import theano.tensor as T
import scipy

import data
import utils
import nn_plankton


CONFIGS = ['convroll4_doublescale_fs5', 'cp8', 'convroll4_big_wd_maxout512',
           'triplescale_fs2_fs5', 'cr4_ds', 'convroll5_preinit_resume_drop@420',
           'doublescale_fs5_latemerge_2233', 'convroll_all_broaden_7x7_weightdecay', 'convroll4_1024_lesswd',
           'convroll4_big_weightdecay']

BAGGED_CONFIGS = ['convroll4_doublescale_fs5', 'cp8', 'convroll4_big_wd_maxout512',
                  'cr4_ds', 'convroll5_preinit_resume_drop@420',
                  'convroll_all_broaden_7x7_weightdecay', 'convroll4_big_weightdecay']

# creating and checking the paths
n_models = len(CONFIGS)
valid_predictions_paths = []
for config in CONFIGS:
    p = 'predictions/valid--blend_featblend_%s--featblend_%s--avg-prob.npy' % (config, config)
    valid_predictions_paths.append(p)

test_predictions_paths = [p.replace('valid--', 'test--', 1) for p in valid_predictions_paths]
test_bagged_prediction_paths = []
for bagged_config in BAGGED_CONFIGS:
    bagged_p = 'predictions/bagged--test--blend_featblend_bagged_%s--avg-prob.npy' % bagged_config
    test_bagged_prediction_paths.append(bagged_p)
    for i in xrange(n_models):
        if bagged_config in test_predictions_paths[i]:
            test_predictions_paths[i] = bagged_p

test_unbagged_prediction_paths = [p for p in test_predictions_paths if 'bagged' not in p]

missing_predictions = []
for path in valid_predictions_paths + test_bagged_prediction_paths + test_unbagged_prediction_paths:
    if not os.path.isfile(path):
        missing_predictions.append(path)

if missing_predictions:
    print '\tPlease generate the following predictions:\n\t%s' % '\n\t'.join(missing_predictions)
    sys.exit(0)

# loading validation predictions
s = np.load("validation_split_v1.pkl")
t_valid = data.labels_train[s['indices_valid']]

predictions_list = [np.load(path) for path in valid_predictions_paths]
predictions_stack = np.array(predictions_list).astype(theano.config.floatX)  # num_sources x num_datapoints x 121

print "Individual prediction errors"
individual_prediction_errors = [utils.log_loss(p, t_valid) for p in predictions_list]
del predictions_list
for i in xrange(n_models):
    print individual_prediction_errors[i], os.path.basename(valid_predictions_paths[i])
print

# optimizing weights
X = theano.shared(predictions_stack)  # source predictions
t = theano.shared(utils.one_hot(t_valid))  # targets
W = T.vector('W')

s = T.nnet.softmax(W).reshape((W.shape[0], 1, 1))
weighted_avg_predictions = T.sum(X * s, axis=0)  # T.tensordot(X, s, [[0], [0]])
error = nn_plankton.log_loss(weighted_avg_predictions, t)
grad = T.grad(error, W)

f = theano.function([W], error)
g = theano.function([W], grad)

w_init = np.zeros(n_models, dtype=theano.config.floatX)
out, loss, _ = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)

weights = np.exp(out)
weights /= weights.sum()

print 'Optimal weights'
for i in xrange(n_models):
    print weights[i], os.path.basename(valid_predictions_paths[i])
print

print 'Generating test set predictions'
predictions_list = [np.load(path) for path in test_predictions_paths]
predictions_stack = np.array(predictions_list)  # num_sources x num_datapoints x 121
del predictions_list

target_path = 'predictions/weighted_blend.npy'
weighted_predictions = np.sum(predictions_stack * weights.reshape((weights.shape[0], 1, 1)), axis=0)
np.save(target_path, weighted_predictions)
print ' stored in %s' % target_path
