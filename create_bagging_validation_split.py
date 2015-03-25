import sys
import data
import numpy as np 
import cPickle as pickle
import sklearn.cross_validation

TARGET_PATH_PATTERN = "splits/bagging_split_%d.pkl"

if len(sys.argv) != 2:
    sys.exit("Usage: python create_bagging_validation_split.py <seed>")

seed = int(sys.argv[1])
target_path = TARGET_PATH_PATTERN % seed

split = sklearn.cross_validation.StratifiedShuffleSplit(data.labels_train, n_iter=1, test_size=0.1, random_state=np.random.RandomState(seed))
indices_train, indices_valid = iter(split).next()

with open(target_path, 'w') as f:
    pickle.dump({
        'indices_train': indices_train,
        'indices_valid': indices_valid,
    }, f, pickle.HIGHEST_PROTOCOL)

print "Split stored in %s" % target_path