import glob
import os

import numpy as np
import skimage.io

directories = glob.glob("data/train/*")
class_names = [os.path.basename(d) for d in directories]
class_names.sort()
num_classes = len(class_names)

paths_train = glob.glob("data/train/*/*")
paths_train.sort()

paths_test = glob.glob("data/test/*")
paths_test.sort()

paths = {
    'train': paths_train,
    'test': paths_test,
}

labels_train = np.zeros(len(paths['train']), dtype='int32')
for k, path in enumerate(paths['train']):
    class_name = os.path.basename(os.path.dirname(path))
    labels_train[k] = class_names.index(class_name)


def load(subset='train'):
    """
    Load all images into memory for faster processing
    """
    images = np.empty(len(paths[subset]), dtype='object')
    for k, path in enumerate(paths[subset]):
        img = skimage.io.imread(path, as_grey=True)
        images[k] = img

    return images


print "Saving train labels"
np.save("data/labels_train.npy", labels_train)
print "Gzipping train labels"
os.system("gzip data/labels_train.npy")

print "Loading train images"
images_train = load('train')
print "Saving train images"
np.save("data/images_train.npy", images_train)
del images_train
print "Gzipping train images"
os.system("gzip data/images_train.npy")

print "Loading test images"
images_test = load('test')
np.save("data/images_test.npy", images_test)
del images_test
print "Gzipping test images"
os.system("gzip data/images_test.npy")

print "Done"