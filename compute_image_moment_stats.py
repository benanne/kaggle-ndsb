import cPickle as pickle

import numpy as np
import skimage.measure

import data


target_path_pattern = "data/image_moment_stats_v1_%s.pkl"

for subset in ['train', 'test']:
    print "loading subset %s..." % subset
    d = data.load(subset)

    print "computing image moment statistics for subset %s..." % subset
    centroids = np.zeros((d.shape[0], 2))
    major_axes = np.zeros((d.shape[0],))
    minor_axes = np.zeros((d.shape[0],))
    angles = np.zeros((d.shape[0],))

    for k, im in enumerate(d):
        if k % 1000 == 0:
            print "image %d of %d..." % (k + 1, d.shape[0])

        a = data.uint_to_float(im)

        ms = skimage.measure.moments(a.astype('float64'), order=1)
        x_centroid = ms[1, 0] / ms[0, 0]
        y_centroid = ms[0, 1] / ms[0, 0]

        mc = skimage.measure.moments_central(a.astype('float64'), y_centroid, x_centroid, order=2)
        mu = mc / ms[0, 0]

        mudiff = mu[2, 0] - mu[0, 2]
        angle = 0.5 * np.arctan2(2.0 * mu[1, 1], mudiff) * (180.0 / np.pi)
        if angle < 0.0:
            angle += 180

        covar = np.array([[mu[2, 0], mu[1, 1]], [mu[1, 1], mu[0, 2]]])
        eigvals, eigvecs = np.linalg.eigh(covar)
        majsq = np.max(eigvals)
        minsq = np.min(eigvals)

        major_axis = np.sqrt(majsq)
        minor_axis = np.sqrt(minsq)

        centroids[k, 0] = x_centroid
        centroids[k, 1] = y_centroid
        major_axes[k] = major_axis
        minor_axes[k] = minor_axis
        angles[k] = angle

    target_path = target_path_pattern % subset

    with open(target_path, 'w') as f:
        pickle.dump({
                'centroids': centroids,
                'major_axes': major_axes,
                'minor_axes': minor_axes,
                'angles': angles,
            }, f, pickle.HIGHEST_PROTOCOL)

    print "stored in %s." % target_path
    print
