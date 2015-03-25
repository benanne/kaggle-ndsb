import numpy as np
import glob


files = glob.glob("predictions/test--*sharded*")
if not len(files) == 3:
    print "There should be 3 test-prediction files that match 'predictions/test--*sharded*'. There are %d." % len(files)
    print "The files found:"
    for f in len(files):
        print f

sharded_pl_pred = np.mean([np.load(f) for f in files], 0)
weighted_ensemble = np.load("predictions/weighted_blend.npy")

final_prediction = (sharded_pl_pred + weighted_ensemble) / 2.
np.save("predictions/final_prediction", final_prediction)
