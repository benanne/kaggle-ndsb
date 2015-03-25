import os
import sys
import glob
import numpy as np
import data
import utils
import os.path


if not len(sys.argv) >= 3: 
    sys.exit("fuse_features_bagged.py <config> <metadata transfer>")
else:
    config = sys.argv[1]
    metadata_paths = sys.argv[2:]

times_to_run = 10

assert len(metadata_paths) == times_to_run

subset = "test"

for path in metadata_paths:
    cmd = "python predict_convnet.py %s %s %s"%(config,path,subset)
    print cmd
    os.system(cmd)

predictions_paths = glob.glob("predictions/"+subset+"--"+config+"*")

assert len(predictions_paths) == times_to_run

print "Loading %s set predictions"%subset
predictions_list = [np.load(path) for path in predictions_paths]
predictions_stack = np.array(predictions_list).astype("float32") # num_sources x num_datapoints x 121

uniform_blend = predictions_stack.mean(0)

target_path = "predictions/%s--%s--%s--%s.npy" % ("bagged", subset, "blend_"+config, "avg-prob")
np.save(target_path, uniform_blend)
print "saving in", target_path

