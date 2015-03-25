import sys

import numpy as np
import theano
import theano.tensor as T

import lasagne as nn

import time
import os
import string
import importlib

import utils
import nn_plankton


if not (3 <= len(sys.argv) <= 5):
    sys.exit("Usage: predict_convnet.py <configuration_name> <metadata_path> [subset=test] [avg-method=avg-probs]")

config_name = sys.argv[1]
metadata_path = sys.argv[2]

if len(sys.argv) >= 4:
    subset = sys.argv[3]
else:
    print "no subset specified, predicting for subset 'test'"
    subset = "test"

if len(sys.argv) >= 5:
    avg_method = sys.argv[4]
    supported_methods = ["avg-probs", "avg-logits", "avg-probs-geom", "avg-probs-ent"]
    if avg_method not in supported_methods:
        sys.exit("Averaging method '%s' not recognized. Valid methods are: %s" % (avg_method, supported_methods.join(",")))
else:
    print "no averaging method specified, averaging probabilities"
    avg_method = "avg-probs"


print "Load parameters"
metadata = np.load(metadata_path)
param_values = metadata['param_values']

if config_name == "_":
    config_name = metadata['configuration']

config = importlib.import_module("configurations.%s" % config_name)

filename = os.path.splitext(os.path.basename(metadata_path))[0]
target_path = "predictions/%s--%s--%s--%s.npy" % (subset, config_name, filename, avg_method)

assert metadata['chunks_since_start'] == config.num_chunks_train - 1 # assert that the metadata file contains final parameters.

print "Build model"
l_ins, l_out = config.build_model()[:2]

if avg_method == "avg-logits":
    if not isinstance(l_out, nn_plankton.NonlinLayer):
        sys.exit("ABORTING: the top layer of selected architecture is not a NonlinLayer, so the logits cannot be obtained.")
    l_out = l_out.input_layer # get the logits instead of probabilities

all_layers = nn.layers.get_all_layers(l_out)
num_params = nn.layers.count_params(l_out)
print "  number of parameters: %d" % num_params
print "  layer output shapes:"
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    print "    %s %s" % (name, layer.get_output_shape(),)

output = l_out.get_output(deterministic=True)

if avg_method == "avg-probs-geom":
    output = T.log(output)

input_ndims = [len(l_in.get_output_shape()) for l_in in l_ins]
xs_shared = [nn.utils.shared_empty(dim=ndim) for ndim in input_ndims]

idx = T.lscalar('idx')

givens = {}
for l_in, x_shared in zip(l_ins, xs_shared):
     givens[l_in.input_var] = x_shared[idx*config.batch_size:(idx+1)*config.batch_size]

compute_output = theano.function([idx], output, givens=givens, on_unused_input='ignore')

nn.layers.set_all_param_values(l_out, param_values)


print "Load data"
config.data_loader.set_params(metadata['data_loader_params'])
# don't call config.data_loader.estimate_params() here! Parameters don't need to be estimated.

augment = not subset.endswith("noaug")
if subset.startswith("test"):
    config.data_loader.load_test()
    if hasattr(config, 'create_eval_test_gen'):
        gen = config.create_eval_test_gen()
        images = config.data_loader.images_test
    else:
        images = config.data_loader.images_test
        gen = config.data_loader.create_fixed_gen(images, augment=augment)
elif subset.startswith("valid"):
    config.data_loader.load_train() # validation set is a subset of the training data
    if hasattr(config, 'create_eval_valid_gen'):
        gen = config.create_eval_valid_gen()
    else:
        images = config.data_loader.images_valid
        gen = config.data_loader.create_fixed_gen(images, augment=augment)
elif subset.startswith("train"):
    config.data_loader.load_train() # train set is a subset of the training data
    if hasattr(config, 'create_eval_train_gen'):
        gen = config.create_eval_train_gen()
    else:
        images = config.data_loader.images_train
        gen = config.data_loader.create_fixed_gen(images, augment=augment)
else:
    print "Unknown subset: %s" % subset



if augment:
    print "  using test-time augmentation"
    num_test_tfs = len(config.data_loader.augmentation_transforms_test)
    # num_predictions = len(images) * num_test_tfs
else:
    print "  NOT using test-time augmentation (noaug)"
    # num_predictions = len(images)

# print "  %d predictions will be made" % num_predictions
# print "  number of chunks: %d" % int(np.ceil(num_predictions / float(config.chunk_size)))
# print


print "Compute output"
num_batches_chunk = config.chunk_size // config.batch_size

outputs = []
remainder = None
for e, (xs_chunk, chunk_length) in enumerate(gen):
    num_batches_chunk = int(np.ceil(chunk_length / float(config.batch_size)))

    print "Chunk %d" % (e + 1)

    print "  load data onto GPU"
    for x_shared, x_chunk in zip(xs_shared, xs_chunk):
        x_shared.set_value(x_chunk)

    print "  compute output in batches"
    outputs_chunk = []
    for b in xrange(num_batches_chunk):
        out = compute_output(b)
        outputs_chunk.append(out)

    outputs_chunk = np.vstack(outputs_chunk)
    outputs_chunk = outputs_chunk[:chunk_length] # truncate to the right length

    if augment and num_test_tfs > 1:
        print "  average over augmentation transforms"
        if remainder is not None: # tack on the remainder from the previous iteration
            outputs_chunk = np.vstack([remainder, outputs_chunk])

        l = (outputs_chunk.shape[0] // num_test_tfs) * num_test_tfs
        remainder = outputs_chunk[l:] # new remainder

        if avg_method == "avg-probs-ent": # entropy-weighted averaging
            outputs_chunk = outputs_chunk[:l]
            h = utils.entropy(outputs_chunk)
            outputs_chunk *= np.exp(-h)[:, None]
            outputs_chunk = outputs_chunk.reshape(l // num_test_tfs, num_test_tfs, outputs_chunk.shape[1]).sum(1)
            z = np.exp(-h).reshape(l // num_test_tfs, num_test_tfs).sum(1)
            outputs_chunk /= z[:, None]
        else:
            outputs_chunk = outputs_chunk[:l].reshape(l // num_test_tfs, num_test_tfs, outputs_chunk.shape[1]).mean(1)

    outputs.append(outputs_chunk)

assert (remainder is None) or remainder.size == 0 # make sure we haven't left any predictions behind
outputs = np.vstack(outputs)


if avg_method == "avg-logits":
    print "Passing averaged logits through the softmax"
    outputs = utils.softmax(outputs)
elif avg_method == "avg-probs-geom":
    print "Renormalizing geometrically averaged probabilities"
    outputs = utils.softmax(outputs)


print "Saving"
np.save(target_path, outputs)
print "  saved to %s" % target_path
    
