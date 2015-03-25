"""
Test-time augmentation tools
"""

import itertools

import numpy as np
import ghalton

import data
import icdf




def build_transforms(**kwargs):
    """
    kwargs are lists of possible values.
    e.g.: build_transforms(rotation=[0, 90, 180, 270], flip=[True, False])

    the names of the arguments are the same as for data.build_augmentation_transform.
    """
    transforms = []

    k = kwargs.keys()
    combinations = list(itertools.product(*kwargs.values()))
    combinations = [dict(zip(k, vals)) for vals in combinations]

    for comb in combinations:
        tf = data.build_augmentation_transform(**comb)
        transforms.append(tf)

    return transforms



def build_quasirandom_transforms(num_transforms, zoom_range, rotation_range, shear_range, translation_range, do_flip=True, allow_stretch=False):
    gen = ghalton.Halton(7)  # 7 dimensions to sample along
    uniform_samples = np.array(gen.get(num_transforms))

    tfs = []
    for s in uniform_samples:
        shift_x = icdf.uniform(s[0], *translation_range)
        shift_y = icdf.uniform(s[1], *translation_range)
        translation = (shift_x, shift_y)

        rotation = icdf.uniform(s[2], *rotation_range)
        shear = icdf.uniform(s[3], *shear_range)

        if do_flip:
            flip = icdf.bernoulli(s[4], p=0.5)
        else:
            flip = False

        log_zoom_range = [np.log(z) for z in zoom_range]
        if isinstance(allow_stretch, float):
            log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
            zoom = np.exp(icdf.uniform(s[5], *log_zoom_range))
            stretch = np.exp(icdf.uniform(s[6], *log_stretch_range))
            zoom_x = zoom * stretch
            zoom_y = zoom / stretch
        elif allow_stretch is True:  # avoid bugs, f.e. when it is an integer
            zoom_x = np.exp(icdf.uniform(s[5], *log_zoom_range))
            zoom_y = np.exp(icdf.uniform(s[6], *log_zoom_range))
        else:
            zoom_x = zoom_y = np.exp(icdf.uniform(s[5], *log_zoom_range))
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

        tfs.append(data.build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip))

    return tfs
