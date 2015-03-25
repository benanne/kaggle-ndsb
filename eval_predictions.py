import sys

import numpy as np

import data
import utils


VALIDATION_SPLIT_PATH = "validation_split_v1.pkl"

if len(sys.argv) != 2:
    sys.exit("Usage: eval_predictions.py <validation_predictions_path>")

path = sys.argv[1]
predictions = np.load(path)

split = np.load(VALIDATION_SPLIT_PATH)
labels_valid = data.labels_train[split['indices_valid']]


loss = utils.log_loss(predictions, labels_valid)
acc = utils.accuracy(predictions, labels_valid)
loss_std = utils.log_loss_std(predictions, labels_valid)

print "Validation loss:\t\t\t%.6f" % loss
print "Classification accuracy:\t\t%.2f%%" % (acc * 100)
print "Validation loss std:\t%.6f" % loss_std
print
for k in xrange(5):
    acc_k = utils.accuracy_topn(predictions, labels_valid, n=k + 1)
    print "Top-%d accuracy:\t\t%.2f%%" % (k + 1, acc_k * 100)