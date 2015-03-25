import os
import sys
import glob
import numpy as np
import data
import utils
import os.path


if not len(sys.argv) == 3: 
    sys.exit("blend_momentsinfo_haralick.py <paths>")
else:
    paths = sys.argv[1:3]


for p in paths: print p

if "test" in  paths[0]:
    subset="test"
elif "valid" in  paths[0]:
    subset="valid"

print
print  subset
print


predictions_list = [np.load(path) for path in paths]
predictions_stack = np.array(predictions_list).astype("float32") # num_sources x num_datapoints x 121
uniform_blend = predictions_stack.mean(0)

print paths[0]

target_path = paths[0].replace("haralick","blend").replace("momentsinfo","blend")

# target_path = "predictions/%s--%s--%s--%s.npy" % (subset, "blend_"+config, config, "avg-prob")
if os.path.isfile(target_path):
    sys.exit("file %s already exists"%target_path)

if subset=="valid":
    t_valid = data.labels_train[np.load("validation_split_v1.pkl")['indices_valid']]
    loss_uniform_selected = utils.log_loss(uniform_blend, t_valid)
    print
    print "%s score: %.6f"%(subset,loss_uniform_selected)
    print
    
np.save(target_path, uniform_blend)
print "saving in", target_path


# ["valid--blend_featharalick_convroll4_1024_lesswd--featharalick_convroll4_1024_lesswd--avg-prob.npy"
# ,"valid--blend_featharalick_convroll4_big_wd_maxout512--featharalick_convroll4_big_wd_maxout512--avg-prob.npy"
# ,"valid--blend_featharalick_convroll4_big_weightdecay--featharalick_convroll4_big_weightdecay--avg-prob.npy"
# ,"valid--blend_featharalick_convroll4_doublescale_fs5--featharalick_convroll4_doublescale_fs5--avg-prob.npy"
# ,"valid--blend_featharalick_convroll5_preinit_resume_drop@420--featharalick_convroll5_preinit_resume_drop@420--avg-prob.npy"
# ,"valid--blend_featharalick_convroll_all_broaden_7x7_weightdecay_resume--featharalick_convroll_all_broaden_7x7_weightdecay_resume--avg-prob.npy"
# ,"valid--blend_featharalick_cp8--featharalick_cp8--avg-prob.npy"
# ,"valid--blend_featharalick_cr4_ds_4stage_big--featharalick_cr4_ds_4stage_big--avg-prob.npy"
# ,"valid--blend_featharalick_triplescale_fs2_fs5--featharalick_triplescale_fs2_fs5--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_convroll4_1024_lesswd--featmomentsinfo_convroll4_1024_lesswd--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_convroll4_big_wd_maxout512--featmomentsinfo_convroll4_big_wd_maxout512--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_convroll4_big_weightdecay--featmomentsinfo_convroll4_big_weightdecay--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_convroll4_doublescale_fs5--featmomentsinfo_convroll4_doublescale_fs5--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_convroll5_preinit_resume_drop@420--featmomentsinfo_convroll5_preinit_resume_drop@420--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_convroll_all_broaden_7x7_weightdecay_resume--featmomentsinfo_convroll_all_broaden_7x7_weightdecay_resume--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_cp8--featmomentsinfo_cp8--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_cr4_ds_4stage_big--featmomentsinfo_cr4_ds_4stage_big--avg-prob.npy"
# ,"valid--blend_featmomentsinfo_triplescale_fs2_fs5--featmomentsinfo_triplescale_fs2_fs5--avg-prob.npy"]