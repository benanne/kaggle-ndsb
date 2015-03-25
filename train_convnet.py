import numpy as np
import theano
import theano.tensor as T

import lasagne as nn

import time
import os
import sys
import importlib
import cPickle as pickle
from datetime import datetime, timedelta
import string
from itertools import izip

import matplotlib
matplotlib.use('agg')
import pylab as plt

import data
import utils
import buffering
import nn_plankton

from subprocess import Popen


if len(sys.argv) < 2:
    sys.exit("Usage: train_convnet.py <configuration_name>")

config_name = sys.argv[1]
config = importlib.import_module("configurations.%s" % config_name)


expid = utils.generate_expid(config_name)
metadata_tmp_path = "/var/tmp/%s.pkl" % expid
metadata_target_path = os.path.join(os.getcwd(), "metadata/%s.pkl" % expid)

print
print "Experiment ID: %s" % expid
print

print "Build model"
model = config.build_model()
if len(model) == 4:
    l_ins, l_out, l_resume, l_exclude = model
elif len(model) == 3:
    l_ins, l_out, l_resume = model
    l_exclude = l_ins[0]
else:
    l_ins, l_out = model
    l_resume = l_out
    l_exclude = l_ins[0]


all_layers = nn.layers.get_all_layers(l_out)
num_params = nn.layers.count_params(l_out)
print "  number of parameters: %d" % num_params
print "  layer output shapes:"
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    print "    %s %s" % (name, layer.get_output_shape(),)

if hasattr(config, 'build_objective'):
    obj = config.build_objective(l_ins, l_out)
else:
    obj = nn.objectives.Objective(l_out, loss_function=nn_plankton.log_loss)


train_loss = obj.get_loss()
output = l_out.get_output(deterministic=True)

all_params = nn.layers.get_all_params(l_out)
all_excluded_params = nn.layers.get_all_params(l_exclude)
all_params = list(set(all_params) - set(all_excluded_params))

input_ndims = [len(l_in.get_output_shape()) for l_in in l_ins]
xs_shared = [nn.utils.shared_empty(dim=ndim) for ndim in input_ndims]
y_shared = nn.utils.shared_empty(dim=2)

if hasattr(config, 'learning_rate_schedule'):
    learning_rate_schedule = config.learning_rate_schedule
else:
    learning_rate_schedule = { 0: config.learning_rate }
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

idx = T.lscalar('idx')

givens = {
    obj.target_var: y_shared[idx*config.batch_size:(idx+1)*config.batch_size],
}
for l_in, x_shared in zip(l_ins, xs_shared):
     givens[l_in.input_var] = x_shared[idx*config.batch_size:(idx+1)*config.batch_size]


if hasattr(config, 'build_updates'):
    updates = config.build_updates(train_loss, all_params, learning_rate)
else:
    updates = nn.updates.nesterov_momentum(train_loss, all_params, learning_rate, config.momentum)

if hasattr(config, 'censor_updates'):
    updates = config.censor_updates(updates, l_out)


iter_train = theano.function([idx], train_loss, givens=givens, updates=updates)
compute_output = theano.function([idx], output, givens=givens, on_unused_input="ignore")


if hasattr(config, 'resume_path'):
    print "Load model parameters for resuming"
    if hasattr(config, 'pre_init_path'):
        print "lresume=lout"
        l_resume = l_out
    resume_metadata = np.load(config.resume_path)
    nn.layers.set_all_param_values(l_resume, resume_metadata['param_values'])

    start_chunk_idx = resume_metadata['chunks_since_start'] + 1
    chunks_train_idcs = range(start_chunk_idx, config.num_chunks_train)

    # set lr to the correct value
    current_lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
    print "  setting learning rate to %.7f" % current_lr
    learning_rate.set_value(current_lr)
    losses_train = resume_metadata['losses_train']
    losses_eval_valid = resume_metadata['losses_eval_valid']
    losses_eval_train = resume_metadata['losses_eval_train']
elif hasattr(config, 'pre_init_path'):
    print "Load model parameters for initializing first x layers"
    resume_metadata = np.load(config.pre_init_path)
    nn.layers.set_all_param_values(l_resume, resume_metadata['param_values'][-len(all_excluded_params):])

    chunks_train_idcs = range(config.num_chunks_train)
    losses_train = []
    losses_eval_valid = []
    losses_eval_train = []
else:
    chunks_train_idcs = range(config.num_chunks_train)
    losses_train = []
    losses_eval_valid = []
    losses_eval_train = []


print "Load data"
config.data_loader.load_train()

if hasattr(config, 'resume_path'):
    config.data_loader.set_params(resume_metadata['data_loader_params'])
else:
    config.data_loader.estimate_params() # important! this takes care of zmuv parameter estimation etc.


if hasattr(config, 'create_train_gen'):
    create_train_gen = config.create_train_gen
else:
    create_train_gen = lambda: config.data_loader.create_random_gen(config.data_loader.images_train, config.data_loader.labels_train)

if hasattr(config, 'create_eval_valid_gen'):
    create_eval_valid_gen = config.create_eval_valid_gen
else:
    create_eval_valid_gen = lambda: config.data_loader.create_fixed_gen(config.data_loader.images_valid, augment=False)

if hasattr(config, 'create_eval_train_gen'):
    create_eval_train_gen = config.create_eval_train_gen
else:
    create_eval_train_gen = lambda: config.data_loader.create_fixed_gen(config.data_loader.images_train, augment=False)


print "Train model"
start_time = time.time()
prev_time = start_time

copy_process = None

num_batches_chunk = config.chunk_size // config.batch_size

for e, (xs_chunk, y_chunk) in izip(chunks_train_idcs, create_train_gen()):
    print "Chunk %d/%d" % (e + 1, config.num_chunks_train)

    if e in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[e])
        print "  setting learning rate to %.7f" % lr
        learning_rate.set_value(lr)

    print "  load training data onto GPU"
    for x_shared, x_chunk in zip(xs_shared, xs_chunk):
        x_shared.set_value(x_chunk)
    y_shared.set_value(y_chunk)

    print "  batch SGD"
    losses = []
    for b in xrange(num_batches_chunk):
        loss = iter_train(b)
        if np.isnan(loss):
            raise RuntimeError("NaN DETECTED.")
        losses.append(loss)

        
    mean_train_loss = np.mean(losses)
    print "  mean training loss:\t\t%.6f" % mean_train_loss
    losses_train.append(mean_train_loss)

    if ((e + 1) % config.validate_every) == 0:
        print
        print "Validating"
        subsets = ["train", "valid"]
        gens = [create_eval_train_gen, create_eval_valid_gen]
        label_sets = [config.data_loader.labels_train, config.data_loader.labels_valid]
        losses_eval = [losses_eval_train, losses_eval_valid]

        for subset, create_gen, labels, losses in zip(subsets, gens, label_sets, losses_eval):
            print "  %s set" % subset
            outputs = []
            for xs_chunk_eval, chunk_length_eval in create_gen():
                num_batches_chunk_eval = int(np.ceil(chunk_length_eval / float(config.batch_size)))

                for x_shared, x_chunk_eval in zip(xs_shared, xs_chunk_eval):
                    x_shared.set_value(x_chunk_eval)

                outputs_chunk = []
                for b in xrange(num_batches_chunk_eval):
                    out = compute_output(b)
                    outputs_chunk.append(out)

                outputs_chunk = np.vstack(outputs_chunk)
                outputs_chunk = outputs_chunk[:chunk_length_eval] # truncate to the right length
                outputs.append(outputs_chunk)

            outputs = np.vstack(outputs)
            loss = utils.log_loss(outputs, labels)
            acc = utils.accuracy(outputs, labels)
            print "    loss:\t%.6f" % loss
            print "    acc:\t%.2f%%" % (acc * 100)
            print

            losses.append(loss)
            del outputs


    now = time.time()
    time_since_start = now - start_time
    time_since_prev = now - prev_time
    prev_time = now
    est_time_left = time_since_start * (float(config.num_chunks_train - (e + 1)) / float(e + 1 - chunks_train_idcs[0]))
    eta = datetime.now() + timedelta(seconds=est_time_left)
    eta_str = eta.strftime("%c")
    print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
    print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
    print

    if ((e + 1) % config.save_every) == 0:
        print
        print "Saving metadata, parameters"

        with open(metadata_tmp_path, 'w') as f:
            pickle.dump({
                'configuration': config_name,
                'experiment_id': expid,
                'chunks_since_start': e,
                'losses_train': losses_train,
                'losses_eval_valid': losses_eval_valid,
                'losses_eval_train': losses_eval_train,
                'time_since_start': time_since_start,
                'param_values': nn.layers.get_all_param_values(l_out), 
                'data_loader_params': config.data_loader.get_params(),
            }, f, pickle.HIGHEST_PROTOCOL)

        # terminate the previous copy operation if it hasn't finished
        if copy_process is not None:
            copy_process.terminate()

        copy_process = Popen(['cp', metadata_tmp_path, metadata_target_path])

        print "  saved to %s, copying to %s" % (metadata_tmp_path, metadata_target_path)
        print
