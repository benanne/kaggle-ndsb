import numpy as np
import itertools

import data
import buffering
import utils


DEFAULT_VALIDATION_SPLIT_PATH = "validation_split_v1.pkl"

class PredictionsWithFeaturesDataLoader(object):
    params = [] # attributes that need to be stored after training and loaded at test time.

    def __init__(self, **kwargs):
        self.augmentation_transforms_test = [data.tform_identity] # default to no test-time augmentation
        self.__dict__.update(kwargs)

    def estimate_params(self):
        pass

    def load_train(self):
        labels = utils.one_hot(data.labels_train, m=121).astype(np.float32)
        split = np.load(DEFAULT_VALIDATION_SPLIT_PATH)

        split = np.load(DEFAULT_VALIDATION_SPLIT_PATH)
        indices_train = split['indices_train']
        indices_valid = split['indices_valid']
        features = np.load("data/features_train.pkl").item()

        if "aaronmoments" in self.features:
            print "aaronmoments"
            def normalize(x):
                return x
                # return (x - x.mean(axis=0,keepdims=True))/x.std(axis=0,keepdims=True)
            image_shapes = np.asarray([img.shape for img in data.load('train')]).astype(np.float32)
            moments = np.load("data/image_moment_stats_v1_train.pkl")
            centroid_distance = np.abs(moments["centroids"][:, [1, 0]] - image_shapes / 2)
            angles = moments["angles"][:, None]
            minor_axes = moments["minor_axes"][:, None]
            major_axes = moments["major_axes"][:, None]
            centroid_distance = normalize(centroid_distance)
            angles = normalize(angles)
            minor_axes = normalize(minor_axes)
            major_axes = normalize(major_axes)
            features["aaronmoments"] = np.concatenate([centroid_distance,angles,minor_axes,major_axes], 1).astype(np.float32)

        info = np.concatenate([features[feat] for feat in self.features], 1).astype(np.float32)

        print info.shape

        self.info_train = info[indices_train]
        self.info_valid = info[indices_valid]

        self.y_train = np.load(self.train_pred_file).astype(np.float32)
        self.y_valid = np.load(self.valid_pred_file).astype(np.float32)
        self.labels_train = labels[indices_train]
        self.labels_valid = labels[indices_valid]

    def load_test(self):
        self.y_test = np.load(self.test_pred_file).astype(np.float32)
        self.images_test = data.load('test')
        features = np.load("data/features_test.pkl").item()

        if "aaronmoments" in self.features:
            print "aaronmoments"
            def normalize(x):
                return x
                # return (x - x.mean(axis=0,keepdims=True))/x.std(axis=0,keepdims=True)
            image_shapes = np.asarray([img.shape for img in self.images_test]).astype(np.float32)
            moments = np.load("data/image_moment_stats_v1_test.pkl")
            centroid_distance = np.abs(moments["centroids"][:, [1, 0]] - image_shapes / 2)
            angles = moments["angles"][:, None]
            minor_axes = moments["minor_axes"][:, None]
            major_axes = moments["major_axes"][:, None]
            centroid_distance = normalize(centroid_distance)
            angles = normalize(angles)
            minor_axes = normalize(minor_axes)
            major_axes = normalize(major_axes)
            features["aaronmoments"] = np.concatenate([centroid_distance,angles,minor_axes,major_axes], 1).astype(np.float32)

        self.info_test = np.concatenate([features[feat] for feat in self.features], 1).astype(np.float32)


    def create_random_gen(self):
        def random_gen():
            for i in range(self.num_chunks_train):
                indices = np.random.randint(self.y_train.shape[0], size=self.chunk_size)
                yield [self.y_train[indices], self.info_train[indices]], self.labels_train[indices]

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, subset):
        if subset == "train":
            y = self.y_train
            image_shapes = self.info_train
        elif subset == "valid":
            y = self.y_valid
            image_shapes = self.info_valid
        elif subset == "test":
            y = self.y_test
            image_shapes = self.info_test
        else:
            raise Exception

        num_batches = int(np.ceil(float(y.shape[0]) / self.chunk_size))
        def fixed_gen():
            for i in range(num_batches):
                if i == num_batches - 1:
                    chunk_x1 = np.zeros((self.chunk_size, y.shape[1]), dtype=np.float32)
                    chunk_x2 = np.zeros((self.chunk_size, image_shapes.shape[1]), dtype=np.float32)
                    chunk_length = y.shape[0] - (num_batches - 1) * self.chunk_size
                    chunk_x1[:chunk_length] = y[i * self.chunk_size:]
                    chunk_x2[:chunk_length] = image_shapes[i * self.chunk_size:]
                else:
                    chunk_x1 = y[i * self.chunk_size: (i + 1) * self.chunk_size]
                    chunk_x2 = image_shapes[i * self.chunk_size: (i + 1) * self.chunk_size]
                    chunk_length = self.chunk_size
                yield [chunk_x1, chunk_x2], chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())

    def get_params(self):
        return { pname: getattr(self, pname, None) for pname in self.params }
        
    def set_params(self, p):
        self.__dict__.update(p)


class PredictionsWithMomentsDataLoader(object):
    params = [] # attributes that need to be stored after training and loaded at test time.

    def __init__(self, **kwargs):
        self.augmentation_transforms_test = [data.tform_identity] # default to no test-time augmentation
        self.__dict__.update(kwargs)

    def estimate_params(self):
        pass

    def load_train(self):
        labels = utils.one_hot(data.labels_train, m=121).astype(np.float32)
        split = np.load(DEFAULT_VALIDATION_SPLIT_PATH)

        split = np.load(DEFAULT_VALIDATION_SPLIT_PATH)
        indices_train = split['indices_train']
        indices_valid = split['indices_valid']

        image_shapes = np.asarray([img.shape for img in data.load('train')]).astype(np.float32)
        moments = np.load("data/image_moment_stats_v1_train.pkl")

        centroid_distance = np.abs(moments["centroids"][:, [1, 0]] - image_shapes / 2)
        info = np.concatenate((centroid_distance, image_shapes, moments["angles"][:, None], moments["minor_axes"][:, None], moments["major_axes"][:, None]), 1).astype(np.float32)

        self.info_train = info[indices_train]
        self.info_valid = info[indices_valid]

        self.y_train = np.load(self.train_pred_file).astype(np.float32)
        self.y_valid = np.load(self.valid_pred_file).astype(np.float32)
        self.labels_train = labels[indices_train]
        self.labels_valid = labels[indices_valid]

    def load_test(self):
        self.y_test = np.load(self.test_pred_file).astype(np.float32)
        self.images_test = data.load('test')
        image_shapes_test = np.asarray([img.shape for img in self.images_test]).astype(np.float32)
        moments_test = np.load("data/image_moment_stats_v1_test.pkl")
        centroid_distance = np.abs(moments_test["centroids"][:, [1, 0]] - image_shapes_test / 2)
        self.info_test = np.concatenate((centroid_distance, image_shapes_test, moments_test["angles"][:, None], moments_test["minor_axes"][:, None], moments_test["major_axes"][:, None]), 1).astype(np.float32)
        # self.info_test = np.concatenate((image_shapes_test, moments_test["centroids"], moments_test["minor_axes"][:, None], moments_test["major_axes"][:, None]), 1).astype(np.float32)


    def create_random_gen(self):
        def random_gen():
            for i in range(self.num_chunks_train):
                indices = np.random.randint(self.y_train.shape[0], size=self.chunk_size)
                yield [self.y_train[indices], self.info_train[indices]], self.labels_train[indices]

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, subset):
        if subset == "train":
            y = self.y_train
            image_shapes = self.info_train
        elif subset == "valid":
            y = self.y_valid
            image_shapes = self.info_valid
        elif subset == "test":
            y = self.y_test
            image_shapes = self.info_test
        else:
            raise Exception

        num_batches = int(np.ceil(float(y.shape[0]) / self.chunk_size))
        def fixed_gen():
            for i in range(num_batches):
                if i == num_batches - 1:
                    chunk_x1 = np.zeros((self.chunk_size, y.shape[1]), dtype=np.float32)
                    chunk_x2 = np.zeros((self.chunk_size, image_shapes.shape[1]), dtype=np.float32)
                    chunk_length = y.shape[0] - (num_batches - 1) * self.chunk_size
                    chunk_x1[:chunk_length] = y[i * self.chunk_size:]
                    chunk_x2[:chunk_length] = image_shapes[i * self.chunk_size:]
                else:
                    chunk_x1 = y[i * self.chunk_size: (i + 1) * self.chunk_size]
                    chunk_x2 = image_shapes[i * self.chunk_size: (i + 1) * self.chunk_size]
                    chunk_length = self.chunk_size
                yield [chunk_x1, chunk_x2], chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())

    def get_params(self):
        return { pname: getattr(self, pname, None) for pname in self.params }
        
    def set_params(self, p):
        self.__dict__.update(p)


class PredictionsWithSizeDataLoader(object):
    params = [] # attributes that need to be stored after training and loaded at test time.

    def __init__(self, **kwargs):
        self.augmentation_transforms_test = [data.tform_identity] # default to no test-time augmentation
        self.__dict__.update(kwargs)

    def estimate_params(self):
        pass

    def load_train(self):
        labels = utils.one_hot(data.labels_train, m=121).astype(np.float32)
        split = np.load(DEFAULT_VALIDATION_SPLIT_PATH)

        split = np.load(DEFAULT_VALIDATION_SPLIT_PATH)
        indices_train = split['indices_train']
        indices_valid = split['indices_valid']

        image_shapes = np.asarray([img.shape for img in data.load('train')]).astype(np.float32)
        self.image_shapes_train = image_shapes[indices_train]
        self.image_shapes_valid = image_shapes[indices_valid]

        self.y_train = np.load(self.train_pred_file).astype(np.float32)
        self.y_valid = np.load(self.valid_pred_file).astype(np.float32)
        self.labels_train = labels[indices_train]
        self.labels_valid = labels[indices_valid]

    def load_test(self):
        self.y_test = np.load(self.test_pred_file).astype(np.float32)
        self.images_test = data.load('test')
        self.image_shapes_test = np.asarray([img.shape for img in self.images_test]).astype(np.float32)

    def create_random_gen(self):
        def random_gen():
            for i in range(self.num_chunks_train):
                indices = np.random.randint(self.y_train.shape[0], size=self.chunk_size)
                yield [self.y_train[indices], self.image_shapes_train[indices]], self.labels_train[indices]

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, subset):
        if subset == "train":
            y = self.y_train
            image_shapes = self.image_shapes_train
        elif subset == "valid":
            y = self.y_valid
            image_shapes = self.image_shapes_valid
        elif subset == "test":
            y = self.y_test
            image_shapes = self.image_shapes_test
        else:
            raise Exception

        num_batches = int(np.ceil(float(y.shape[0]) / self.chunk_size))
        def fixed_gen():
            for i in range(num_batches):
                if i == num_batches - 1:
                    chunk_x1 = np.zeros((self.chunk_size, y.shape[1]), dtype=np.float32)
                    chunk_x2 = np.zeros((self.chunk_size, image_shapes.shape[1]), dtype=np.float32)
                    chunk_length = y.shape[0] - (num_batches - 1) * self.chunk_size
                    chunk_x1[:chunk_length] = y[i * self.chunk_size:]
                    chunk_x2[:chunk_length] = image_shapes[i * self.chunk_size:]
                else:
                    chunk_x1 = y[i * self.chunk_size: (i + 1) * self.chunk_size]
                    chunk_x2 = image_shapes[i * self.chunk_size: (i + 1) * self.chunk_size]
                    chunk_length = self.chunk_size
                yield [chunk_x1, chunk_x2], chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())

    def get_params(self):
        return { pname: getattr(self, pname, None) for pname in self.params }
        
    def set_params(self, p):
        self.__dict__.update(p)



class DataLoader(object):
    params = [] # attributes that need to be stored after training and loaded at test time.

    def __init__(self, **kwargs):
        self.augmentation_transforms_test = [data.tform_identity] # default to no test-time augmentation
        self.__dict__.update(kwargs)

        if not hasattr(self, 'validation_split_path'):
            self.validation_split_path = DEFAULT_VALIDATION_SPLIT_PATH
            print "using default validation split: %s" % self.validation_split_path
        else:
            print "using NON-default validation split: %s" % self.validation_split_path

    def estimate_params(self):
        pass

    def load_train(self):
        images = data.load('train')
        labels = utils.one_hot(data.labels_train, m=121).astype(np.float32)

        split = np.load(self.validation_split_path)
        indices_train = split['indices_train']
        indices_valid = split['indices_valid']

        self.images_train = images[indices_train]
        self.labels_train = labels[indices_train]
        self.images_valid = images[indices_valid]
        self.labels_valid = labels[indices_valid]

    def load_test(self):
        self.images_test = data.load('test')

    def get_params(self):
        return { pname: getattr(self, pname, None) for pname in self.params }
        
    def set_params(self, p):
        self.__dict__.update(p)


class RescaledDataLoader(DataLoader):
    def create_random_gen(self, images, labels):
        gen = data.rescaled_patches_gen_augmented(images, labels, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, num_chunks=self.num_chunks_train, augmentation_params=self.augmentation_params)

        def random_gen():
            for chunk_x, chunk_y, chunk_shape in gen:
                yield [chunk_x[:, None, :, :]], chunk_y

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, images, augment=False):
        augmentation_transforms = self.augmentation_transforms_test if augment else None
        gen = data.rescaled_patches_gen_fixed(images, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, augmentation_transforms=augmentation_transforms)
        
        def fixed_gen():
            for chunk_x, chunk_shape, chunk_length in gen:
                yield [chunk_x[:, None, :, :]], chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())


class ZmuvRescaledDataLoader(RescaledDataLoader):
    params = ['zmuv_mean', 'zmuv_std'] # params that need to be stored after training and loaded at test time.

    def estimate_params(self):
        self.estimate_zmuv_params() # zero mean unit variance

    def estimate_zmuv_params(self):
        gen = data.rescaled_patches_gen_augmented(self.images_train, self.labels_train, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, num_chunks=1, augmentation_params=self.augmentation_params)
        chunk_x, _, _ = gen.next()
        self.zmuv_mean = chunk_x.mean()
        self.zmuv_std = chunk_x.std()

    def create_random_gen(self, images, labels):
        gen = data.rescaled_patches_gen_augmented(images, labels, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, num_chunks=self.num_chunks_train, augmentation_params=self.augmentation_params)

        def random_gen():
            for chunk_x, chunk_y, chunk_shape in gen:
                chunk_x -= self.zmuv_mean
                chunk_x /= self.zmuv_std
                yield [chunk_x[:, None, :, :]], chunk_y

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, images, augment=False):
        augmentation_transforms = self.augmentation_transforms_test if augment else None
        gen = data.rescaled_patches_gen_fixed(images, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, augmentation_transforms=augmentation_transforms)
        
        def fixed_gen():
            for chunk_x, chunk_shape, chunk_length in gen:
                chunk_x -= self.zmuv_mean
                chunk_x /= self.zmuv_std 
                yield [chunk_x[:, None, :, :]], chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())


class ZmuvMultiscaleDataLoader(DataLoader):
    params = ['zmuv_means', 'zmuv_stds'] # params that need to be stored after training and loaded at test time.

    def estimate_params(self):
        self.estimate_zmuv_params() # zero mean unit variance

    def estimate_zmuv_params(self):
        gen = data.multiscale_patches_gen_augmented(self.images_train, self.labels_train, self.scale_factors, patch_sizes=self.patch_sizes,
            chunk_size=self.chunk_size, num_chunks=1, augmentation_params=self.augmentation_params)
        chunks_x, _, _ = gen.next()
        self.zmuv_means = [chunk_x.mean() for chunk_x in chunks_x]
        self.zmuv_stds = [chunk_x.std() for chunk_x in chunks_x]

    def create_random_gen(self, images, labels):
        gen = data.multiscale_patches_gen_augmented(images, labels, self.scale_factors, patch_sizes=self.patch_sizes,
            chunk_size=self.chunk_size, num_chunks=self.num_chunks_train, augmentation_params=self.augmentation_params)

        def random_gen():
            for chunks_x, chunk_y, chunk_shape in gen:
                for k in xrange(len(chunks_x)):
                    chunks_x[k] -= self.zmuv_means[k]
                    chunks_x[k] /= self.zmuv_stds[k]
                    chunks_x[k] = chunks_x[k][:, None, :, :]

                yield chunks_x, chunk_y

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, images, augment=False):
        augmentation_transforms = self.augmentation_transforms_test if augment else None
        gen = data.multiscale_patches_gen_fixed(images, self.scale_factors, patch_sizes=self.patch_sizes,
            chunk_size=self.chunk_size, augmentation_transforms=augmentation_transforms)
        
        def fixed_gen():
            for chunks_x, chunk_shape, chunk_length in gen:
                for k in xrange(len(chunks_x)):
                    chunks_x[k] -= self.zmuv_means[k]
                    chunks_x[k] /= self.zmuv_stds[k]
                    chunks_x[k] = chunks_x[k][:, None, :, :]

                yield chunks_x, chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())


class ShardedResampledPseudolabelingZmuvMultiscaleDataLoader(ZmuvMultiscaleDataLoader):

    def load_train(self):
        train_images = data.load('train')
        train_labels = utils.one_hot(data.labels_train).astype(np.float32)

        if ("valid_pred_file" in self.__dict__):
            valid_pseudo_labels = np.load(self.valid_pred_file).astype(np.float32)
        else:
            print "No valid_pred_file set. Only using test-set for pseudolabeling!!"

        shuffle = np.load("test_shuffle_seed0.npy")
        if not ("shard" in self.__dict__):
            raise ValueError("Missing argument: shard: (should be value in {0, 1, 2})")
        if not self.shard in [0, 1, 2]:
            raise ValueError("Wrong argument: shard: (should be value in {0, 1, 2})")
        N = len(shuffle)
        if self.shard == 0:
            train_shard = shuffle[N/3:]
        if self.shard == 1:
            train_shard = np.concatenate((shuffle[:N/3], shuffle[2*N/3:]))
        if self.shard == 2:
            train_shard = shuffle[:2*N/3]

        test_images = data.load('test')[train_shard]
        test_pseudo_labels = np.load(self.test_pred_file)[train_shard].astype(np.float32)
        print test_pseudo_labels.shape

        if not hasattr(self, 'validation_split_path'):
            self.validation_split_path = DEFAULT_VALIDATION_SPLIT_PATH
        split = np.load(self.validation_split_path)
        indices_train = split['indices_train']
        indices_valid = split['indices_valid']

        self.images_train = train_images[indices_train]
        self.labels_train = train_labels[indices_train]
        if ("valid_pred_file" in self.__dict__):
            self.images_pseudo = np.concatenate((train_images[indices_valid], test_images), 0)
            self.labels_pseudo = np.concatenate((valid_pseudo_labels, test_pseudo_labels), 0)
        else:
            self.images_pseudo = test_images
            self.labels_pseudo = test_pseudo_labels

        self.images_valid = train_images[indices_valid]
        self.labels_valid = train_labels[indices_valid]

    def create_random_gen(self, *args):
        # we ignore the args
        train_chunk_size = int(round(self.chunk_size * self.train_sample_weight))
        pseudo_chunk_size = self.chunk_size - train_chunk_size

        train_gen = data.multiscale_patches_gen_augmented(self.images_train, self.labels_train, self.scale_factors, patch_sizes=self.patch_sizes,
            chunk_size=train_chunk_size, num_chunks=self.num_chunks_train, augmentation_params=self.augmentation_params)

        pseudo_gen = data.multiscale_patches_gen_augmented(self.images_pseudo, self.labels_pseudo, self.scale_factors, patch_sizes=self.patch_sizes,
            chunk_size=pseudo_chunk_size, num_chunks=self.num_chunks_train, augmentation_params=self.augmentation_params)

        def random_gen():
            indices = np.arange(self.chunk_size)
            for a, b in itertools.izip(train_gen, pseudo_gen):
                (chunk_x1, chunk_y1, chunk_shape), (chunk_x2, chunk_y2, _) = a, b
                np.random.shuffle(indices)

                chunk_y = np.concatenate((chunk_y1, chunk_y2), 0)[indices]
                chunk_x = []
                for k in xrange(len(chunk_x1)):
                    chunk_x += [np.concatenate((chunk_x1[k], chunk_x2[k]), 0)[indices]]
                    chunk_x[k] -= self.zmuv_means[k]
                    chunk_x[k] /= self.zmuv_stds[k]
                    chunk_x[k] = chunk_x[k][:, None, :, :]
                yield chunk_x, chunk_y

        return buffering.buffered_gen_threaded(random_gen())