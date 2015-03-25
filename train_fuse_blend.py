import os
import sys
import glob
import numpy as np
import data
import utils
import os.path


if not len(sys.argv) in [2,3,4]: 
    sys.exit("train_fuse_blend.py <config> [valid=true] [test=true]")
elif len(sys.argv) == 2:
    gen_test = True
    gen_valid = True
elif len(sys.argv) == 3:
    gen_test = True
    if sys.argv[2] == "valid=true": gen_valid = True
    elif sys.argv[2] == "valid=false": gen_valid = False
    else: sys.exit("train_fuse_blend.py <config> [valid=true] [test=true]")
elif len(sys.argv) == 4:
    if sys.argv[2] == "valid=true": gen_valid = True
    elif sys.argv[2] == "valid=false": gen_valid = False
    else: sys.exit("train_fuse_blend.py <config> [valid=true] [test=true]")

    if sys.argv[3] == "test=true": gen_test = True
    elif sys.argv[3] == "test=false": gen_test = False
    else: sys.exit("train_fuse_blend.py <config> [valid=true] [test=true]")

config = sys.argv[1]

times_to_run = 10

if gen_test: print "generating test predictions"
if gen_valid: print "generating valid predictions"



# Train 10 times
###################

run_paths = glob.glob("metadata/"+config+"-*")

if len(run_paths) < times_to_run:
    for i in range(times_to_run-len(run_paths)):
        cmd = "python train_convnet.py %s" % config
        print cmd
        os.system(cmd)

    run_paths = glob.glob("metadata/"+config+"-*")

print len(run_paths)
assert len(run_paths) == times_to_run


# Make a uniform blend of the trained runs
#########################################################


print "Loading metadata"
run_list = [np.load(path) for path in run_paths]

def generate_pred(subset):

    predictions_paths = glob.glob("predictions/"+subset+"--"+config+"-*")

    if len(predictions_paths) < times_to_run:
        for path in run_paths:
            cmd = "python predict_convnet.py %s %s %s"%(config,path,subset)
            print cmd
            os.system(cmd)

        predictions_paths = glob.glob("predictions/"+subset+"--"+config+"-*")

    assert len(predictions_paths) == times_to_run

    print "Loading %s set predictions"%subset
    predictions_list = [np.load(path) for path in predictions_paths]
    predictions_stack = np.array(predictions_list).astype("float32") # num_sources x num_datapoints x 121

    uniform_blend = predictions_stack.mean(0)

    if subset=="valid":
        t_valid = data.labels_train[np.load("validation_split_v1.pkl")['indices_valid']]
        loss_uniform_selected = utils.log_loss(uniform_blend, t_valid)
        print
        print config
        print "%s score: %.6f"%(subset,loss_uniform_selected)
        print

    target_path = "predictions/%s--%s--%s--%s.npy" % (subset, "blend_"+config, config, "avg-prob")
    np.save(target_path, uniform_blend)
    print "saving in", target_path

if gen_test:
    generate_pred("test")
if gen_valid:
    generate_pred("valid")