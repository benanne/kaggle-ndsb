import data
import numpy as np 
import cPickle as pickle
import sklearn.cross_validation

TARGET_PATH = "validation_split_v1.pkl"

split = sklearn.cross_validation.StratifiedShuffleSplit(data.labels_train, n_iter=1, test_size=0.1, random_state=np.random.RandomState(42))
indices_train, indices_valid = iter(split).next()

with open(TARGET_PATH, 'w') as f:
    pickle.dump({
        'indices_train': indices_train,
        'indices_valid': indices_valid,
    }, f, pickle.HIGHEST_PROTOCOL)

print "Split stored in %s" % TARGET_PATH