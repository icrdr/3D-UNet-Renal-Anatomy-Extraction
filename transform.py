import numpy as np
import scipy.ndimage as ndi


def remove_small_region(input, threshold):
    labels, nb_labels = ndi.label(input)
    label_areas = np.bincount(labels.ravel())
    too_small_labels = label_areas < threshold
    too_small_mask = too_small_labels[labels]
    input[too_small_mask] = 0
    return input


class RemoveSmallRegion(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, case):
        case['label'] = remove_small_region(case['label'], self.threshold)
        return case


def split_dim(input, axis=-1):
    sub_arr = np.split(input, input.shape[axis], axis=axis)
    return [np.squeeze(arr, axis=axis) for arr in sub_arr]


def slice_dim(input, slice, axis=-1):
    return split_dim(input, axis=axis)[slice]


def rescale(input,
            scale,
            order=1,
            mode='reflect',
            cval=0,
            is_label=False,
            multi_class=False):
    '''
    A wrap of scipy.ndimage.zoom for label encoding data support.

    Args:
        See scipy.ndimage.zoom doc rescale for more detail.
        is_label: If true, split label before rescale.
    '''
    dtype = input.dtype

    if is_label:
        num_classes = np.unique(input).max() + 1
    if order == 0 or not is_label or num_classes < 3:
        if multi_class:
            classes = to_tensor(input)
            rescaled_classes = np.array([ndi.zoom(c.astype(np.float32),
                                                  scale,
                                                  order=order,
                                                  mode=mode,
                                                  cval=cval)
                                         for c in classes])
            return to_numpy(rescaled_classes).astype(dtype)
        else:
            return ndi.zoom(input.astype(np.float32),
                            scale,
                            order=order,
                            mode=mode,
                            cval=cval).astype(dtype)
    else:
        onehot = to_one_hot(input, num_classes, to_tensor=True)
        rescaled_onehot = np.array([ndi.zoom(c.astype(np.float32),
                                             scale,
                                             order=order,
                                             mode=mode,
                                             cval=cval)
                                    for c in onehot])
        return np.argmax(rescaled_onehot, axis=0).astype(dtype)


def resize(input,
           shape,
           order=1,
           mode='reflect',
           cval=0,
           is_label=False):
    '''
    Resize ndarray. (wrap of rescale)

    Args:
        See scipy.ndimage.zoom doc rescale for more detail.
        is_label: If true, split label before rescale.
    '''
    orig_shape = input.shape
    multi_class = len(shape) == len(orig_shape)-1
    orig_shape = orig_shape[:len(shape)]
    scale = np.array(shape)/np.array(orig_shape)
    return rescale(input,
                   scale,
                   order=order,
                   mode=mode,
                   cval=cval,
                   is_label=is_label,
                   multi_class=multi_class)


class Resize(object):
    '''
    Resize image and label.

    Args:
        scale (sequence or int): range of factor.
            If it is a int number, the range will be [1-int, 1+int]
    '''

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, case):
        case['image'] = resize(case['image'], self.shape)
        case['label'] = resize(case['label'], self.shape, is_label=True)
        return case


class RandomRescale(object):
    '''
    Randomly rescale image and label by range of scale factor.

    Args:
        scale (sequence or int): range of factor.
            If it is a int number, the range will be [1-int, 1+int]
    '''

    def __init__(self, scale):
        if isinstance(scale, float):
            assert 0 <= scale <= 1, "If range is a single number, it must be non negative"
            self.scale = [1-scale, 1+scale]
        else:
            self.scale = scale

    def __call__(self, case):
        scale = np.random.uniform(self.scale[0], self.scale[1])
        case['image'] = rescale(case['image'], scale)
        case['label'] = rescale(case['label'], scale, is_label=True)
        return case


def to_tensor(input):
    dims_indices = np.arange(len(input.shape))
    dims_indices = np.concatenate((dims_indices[-1:], dims_indices[:-1]))
    return input.transpose(dims_indices)


def to_numpy(input):
    dims_indices = np.arange(len(input.shape))
    dims_indices = np.concatenate((dims_indices[1:], dims_indices[:1]))
    return input.transpose(dims_indices)


class ToTensor(object):
    '''
    (d1,d2,...,dn,class) => (class,d1,d2,...,dn)
    '''

    def __call__(self, case):
        case['image'] = to_tensor(case['image'])
        return case


class ToNumpy(object):
    '''
    (class,d1,d2,...,dn) => (d1,d2,...,dn,class)
    '''

    def __call__(self, case):
        case['image'] = to_numpy(case['image'])
        return case


def adjust_contrast(input, factor):
    dtype = input.dtype
    mean = input.mean()
    return ((input - mean) * factor + mean).astype(dtype)


def adjust_brightness(input, factor):
    dtype = input.dtype
    minimum = input.min()
    return ((input - minimum) * factor + minimum).astype(dtype)


def adjust_gamma(input, gamma, epsilon=1e-7):
    dtype = input.dtype
    minimum = input.min()
    maximum = input.max()
    arange = maximum - minimum + epsilon
    return (np.power(((input - minimum) / arange), gamma) * arange + minimum).astype(dtype)


class RandomContrast(object):
    '''
    Adjust contrast with random factor value in range.

    Args:
        factor (sequence or int): range of factor.
            If it is a int number, the range will be [1-int, 1+int]
    '''

    def __init__(self, factor_range):
        if isinstance(factor_range, float):
            assert 0 <= factor_range <= 1, "If range is a single number, it must be non negative"
            self.factor_range = [1-factor_range, 1+factor_range]
        else:
            self.factor_range = factor_range

    def __call__(self, case):
        factor = np.random.uniform(self.factor_range[0], self.factor_range[1])
        case['image'] = adjust_contrast(case['image'], factor)
        return case


class RandomBrightness(object):
    '''
    Adjust brightness with random factor value in range.

    Args:
        factor_range (sequence or int): range of factor.
            If it is a int number, the range will be [1-int, 1+int]
    '''

    def __init__(self, factor_range):
        if isinstance(factor_range, float):
            assert 0 <= factor_range <= 1, "If range is a single number, it must be non negative"
            self.factor_range = [1-factor_range, 1+factor_range]
        else:
            self.factor_range = factor_range

    def __call__(self, case):
        factor = np.random.uniform(self.factor_range[0], self.factor_range[1])
        case['image'] = adjust_brightness(case['image'], factor)
        return case


class RandomGamma(object):
    '''
    Adjust gamma with random gamma value in range.

    Args:
        gamma_range (sequence or int): range of gamma.
            If it is a int number, the range will be [1-int, 1+int]
    '''

    def __init__(self, gamma_range):
        if isinstance(gamma_range, float):
            assert 0 <= gamma_range <= 1, "If range is a single number, it must be non negative"
            self.gamma_range = [1-gamma_range, 1+gamma_range]
        else:
            self.gamma_range = gamma_range

    def __call__(self, case):
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        case['image'] = adjust_gamma(case['image'], gamma)
        return case


def to_one_hot(input, num_classes, to_tensor=False):
    '''
    Label to one-hot. Label shape changes:
    (d1,d2,...,dn) => (d1,d2,...,dn,class)
    or (d1,d2,...,dn) => (class,d1,d2,...,dn) (pytorch tensor like)

    Args:
        num_classes (int): Total num of label classes.
    '''
    dtype = input.dtype
    onehot = np.eye(num_classes)[input]
    dims_indices = np.arange(len(input.shape)+1)
    if to_tensor:
        dims_indices = np.concatenate((dims_indices[-1:], dims_indices[:-1]))
    return onehot.transpose(dims_indices).astype(dtype)


class RandomMirror(object):
    '''
    Mirroring image and label randomly (per_axis).

    Args:
        p_per_axis (sequence or int): axis u wanted to mirror.
    '''

    def __init__(self, p_per_axis):
        self.p_per_axis = p_per_axis

    def __call__(self, case):
        dim = len(case['image'].shape)-1
        if not isinstance(self.p_per_axis, (np.ndarray, tuple, list)):
            self.p_per_axis = [self.p_per_axis] * dim

        for i, p in enumerate(self.p_per_axis):
            if np.random.uniform() < p:
                # negative strides numpy array is not support for pytorch yet.
                case['image'] = np.flip(case['image'], i).copy()
                case['label'] = np.flip(case['label'], i).copy()

        return case


class ToOnehot(object):
    '''
    Label to one-hot. Label shape changes:
    (d1,d2,...,dn) => (d1,d2,...,dn,class)
    or (d1,d2,...,dn) => (class,d1,d2,...,dn) (with transpose)

    Args:
        num_classes (int): Total num of label classes.
    '''

    def __init__(self, num_classes, to_tensor=False):
        self.num_classes = num_classes
        self.to_tensor = to_tensor

    def __call__(self, case):
        case['label'] = to_one_hot(case['label'], self.num_classes, self.to_tensor)
        return case


def combination_labels(input, combinations, num_classes):
    '''
    Combines some label indices as one.

    Args:
        combinations (ndarray, list, tuple): Combines of label indices
            ndarray, e.g.[[0,1],[2]]
            list, e.g.[0,1]
            tuple, e.g.(0,1)
        num_classes (int): Total num of label classes.
    '''
    dtype = input.dtype
    # add other single class combinations in the combinations setting
    if len(np.array(combinations).shape) == 1:
        combinations = [combinations]
    full_combinations = []
    used_combination_indices = []
    classes_range = range(num_classes)
    for c in classes_range:
        c_pos = np.where(np.array(combinations) == c)
        related_combination_indices = c_pos[0]
        if len(related_combination_indices) > 0:
            for i in related_combination_indices:
                if i not in used_combination_indices:
                    full_combinations.append(combinations[i])
                    used_combination_indices.append(i)
        else:
            full_combinations.append([c])
    onehot = to_one_hot(input, num_classes, True)

    # combination the classes into new onehot
    combination_logics = []
    for combination in full_combinations:
        combination_logic = np.zeros_like(onehot[0])
        for c in combination:
            combination_logic = np.logical_or(onehot[c], combination_logic)
        combination_logics.append(combination_logic)
    combination_logics = np.array(combination_logics)

    # onehot => argmax
    return np.argmax(combination_logics, axis=0).astype(dtype)


class CombineLabels(object):
    '''
    Combines some label indices as one.

    Args:
        combinations (ndarray, list, tuple): Combines of label indices
            ndarray, e.g.[[0,1],[2]]
            list, e.g.[0,1]
            tuple, e.g.(0,1)
        num_classes (int): Total num of label classes.
    '''

    def __init__(self, combinations, num_classes):
        self.combinations = combinations
        self.num_classes = num_classes

    def __call__(self, case):
        case['label'] = combination_labels(case['label'], self.combinations, self.num_classes)
        return case


def pad(input, pad_size, pad_mode='constant', pad_cval=0):
    shape = input.shape
    pad_size = [max(shape[d], pad_size[d]) for d in range(len(pad_size))]
    return crop_pad(input, pad_size, pad_mode=pad_mode, pad_cval=pad_cval)


def crop_pad(input, crop_size, crop_mode='center', crop_margin=0, pad_mode='constant', pad_cval=0):
    dim = len(crop_size)

    if not isinstance(crop_margin, (np.ndarray, tuple, list)):
        crop_margin = [crop_margin] * dim

    bbox = gen_bbox_for_crop(crop_size, input.shape, crop_margin, crop_mode)
    return crop_pad_to_bbox(input, bbox, pad_mode, pad_cval)


def gen_bbox_for_crop(crop_size, orig_shape, crop_margin, crop_mode):
    assert crop_mode == "center" or crop_mode == "random",\
        "crop mode must be either center or random"

    bbox = []
    for i in range(len(orig_shape)):
        if i < len(crop_size):
            if crop_mode == 'random'\
                    and orig_shape[i] - crop_size[i] - crop_margin[i] > crop_margin[i]:
                lower_boundaries = np.random.randint(
                    crop_margin[i], orig_shape[i] - crop_size[i] - crop_margin[i])
            else:
                lower_boundaries = (orig_shape[i] - crop_size[i]) // 2
            bbox.append([lower_boundaries, lower_boundaries+crop_size[i]])
        else:
            bbox.append([0, orig_shape[i]])
    return bbox


def crop_pad_to_bbox(input, bbox, pad_mode='constant', pad_cval=0):
    shape = input.shape
    dtype = input.dtype

    # crop first
    abs_bbox_slice = [slice(max(0, bbox[d][0]), min(bbox[d][1], shape[d]))
                      for d in range(len(shape))]
    cropped = input[tuple(abs_bbox_slice)]

    # than pad
    pad_width = [[abs(min(0, bbox[d][0])), abs(min(0, shape[d] - bbox[d][1]))]
                 for d in range(len(shape))]
    if any([i > 0 for j in pad_width for i in j]):
        cropped = np.pad(cropped, pad_width, pad_mode, constant_values=pad_cval)

    return cropped.astype(dtype)


class Crop(object):
    '''
    Crop image and label simultaneously.

    Args:
        size (sequence or int): The size of crop.
        mode (str): 'random' or 'center'.
        margin (sequence or int): If crop mode is random, it determine how far from
            cropped boundary to shape boundary.
        enforce_label_indices (sequence or int): If crop mode is random, it determine
            the cropped label must contain the setting label index.
        image_pad_mode: np.pad kwargs
        image_pad_cval: np.pad kwargs
        label_pad_mode: np.pad kwargs
        label_pad_cval: np.pad kwargs
    '''

    def __init__(self,
                 crop_size=128,
                 crop_mode='center',
                 crop_margin=0,
                 enforce_label_indices=[],
                 image_pad_mode='constant',
                 image_pad_cval=0,
                 label_pad_mode='constant',
                 label_pad_cval=0):
        self.crop_size = crop_size
        self.crop_mode = crop_mode
        self.crop_margin = crop_margin
        if isinstance(enforce_label_indices, int):
            self.enforce_label_indices = [enforce_label_indices]
        else:
            self.enforce_label_indices = enforce_label_indices
        self.image_pad_mode = image_pad_mode
        self.image_pad_cval = image_pad_cval
        self.label_pad_mode = label_pad_mode
        self.label_pad_cval = label_pad_cval

    def __call__(self, case):
        image, label = case['image'], case['label']
        dim = len(image.shape)-1

        if not isinstance(self.crop_size, (np.ndarray, tuple, list)):
            self.crop_size = [self.crop_size] * dim
        if not isinstance(self.crop_margin, (np.ndarray, tuple, list)):
            self.crop_margin = [self.crop_margin] * dim

        gen_bbox = True
        while gen_bbox:
            bbox = gen_bbox_for_crop(self.crop_size, image.shape, self.crop_margin, self.crop_mode)
            cropped_label = crop_pad_to_bbox(
                label,
                bbox[:-1],
                self.label_pad_mode,
                self.label_pad_cval)
            cropped_label_indices = np.unique(cropped_label)
            gen_bbox = False
            for i in self.enforce_label_indices:
                if i not in cropped_label_indices:
                    # print('cropped label does not contain label %d, regen bbox' % i)
                    gen_bbox = True

        cropped_image = crop_pad_to_bbox(
            image,
            bbox,
            self.image_pad_mode,
            self.image_pad_cval)

        case['image'] = cropped_image
        case['label'] = cropped_label

        return case


class RandomCrop(Crop):
    '''
    Random crop image and label simultaneously.

    Args:
        size (sequence or int): The size of crop.
        margin (sequence or int): It determine how far from cropped boundary to shape boundary
        enforce_label_indices (sequence or int): If crop mode is random, it determine
            the cropped label must contain the setting label index.
        image_pad_mode: np.pad kwargs
        image_pad_cval: np.pad kwargs
        label_pad_mode: np.pad kwargs
        label_pad_cval: np.pad kwargs
    '''

    def __init__(self,
                 crop_size=128,
                 crop_margin=0,
                 enforce_label_indices=[],
                 image_pad_mode='constant',
                 image_pad_cval=0,
                 label_pad_mode='constant',
                 label_pad_cval=0):
        super(RandomCrop, self).__init__(crop_size,
                                         crop_margin=crop_margin,
                                         crop_mode='random',
                                         enforce_label_indices=enforce_label_indices,
                                         image_pad_mode=image_pad_mode,
                                         image_pad_cval=image_pad_cval,
                                         label_pad_mode=label_pad_mode,
                                         label_pad_cval=label_pad_cval)


class CenterCrop(Crop):
    '''
    Center crop image and label simultaneously.

    Args:
        size (sequence or int): The size of crop.
        image_pad_mode: np.pad kwargs
        image_pad_cval: np.pad kwargs
        label_pad_mode: np.pad kwargs
        label_pad_cval: np.pad kwargs
    '''

    def __init__(self,
                 crop_size=128,
                 image_pad_mode='constant',
                 image_pad_cval=0,
                 label_pad_mode='constant',
                 label_pad_cval=0):
        super(CenterCrop, self).__init__(crop_size,
                                         crop_mode='center',
                                         image_pad_mode=image_pad_mode,
                                         image_pad_cval=image_pad_cval,
                                         label_pad_mode=label_pad_mode,
                                         label_pad_cval=label_pad_cval)


class RandomRescaleCrop(Crop):
    '''
    Randomly resize image and label, then crop.

    Args:
        scale (sequence or int): range of factor.
            If it is a int number, the range will be [1-int, 1+int]
    '''

    def __init__(self,
                 scale,
                 crop_size=128,
                 crop_mode='center',
                 crop_margin=0,
                 enforce_label_indices=[],
                 image_pad_mode='constant',
                 image_pad_cval=0,
                 label_pad_mode='constant',
                 label_pad_cval=0):
        super(RandomRescaleCrop, self).__init__(crop_size,
                                                crop_mode=crop_mode,
                                                crop_margin=crop_margin,
                                                enforce_label_indices=enforce_label_indices,
                                                image_pad_mode=image_pad_mode,
                                                image_pad_cval=image_pad_cval,
                                                label_pad_mode=label_pad_mode,
                                                label_pad_cval=label_pad_cval)
        if isinstance(scale, float):
            assert 0 <= scale <= 1, "If range is a single number, it must be non negative"
            self.scale = [1-scale, 1+scale]
        else:
            self.scale = scale

    def __call__(self, case):
        image, label = case['image'], case['label']

        dim = len(image.shape)-1

        if not isinstance(self.crop_size, (np.ndarray, tuple, list)):
            self.crop_size = [self.crop_size] * dim
        if not isinstance(self.crop_margin, (np.ndarray, tuple, list)):
            self.crop_margin = [self.crop_margin] * dim

        scale = np.random.uniform(self.scale[0], self.scale[1])
        crop_size_before_rescale = np.round(np.array(self.crop_size) / scale).astype(np.int)

        # crop first
        gen_bbox = True
        while gen_bbox:
            bbox = gen_bbox_for_crop(crop_size_before_rescale,
                                     image.shape,
                                     self.crop_margin,
                                     self.crop_mode)

            cropped_label = crop_pad_to_bbox(
                label,
                bbox[:-1],
                self.label_pad_mode,
                self.label_pad_cval)

            cropped_label_indices = np.unique(cropped_label)
            gen_bbox = False
            for i in self.enforce_label_indices:
                if i not in cropped_label_indices:
                    # print('cropped label does not contain label %d, regen bbox' % i)
                    gen_bbox = True

        cropped_image = crop_pad_to_bbox(
            image,
            bbox,
            self.image_pad_mode,
            self.image_pad_cval)

        # then resize
        resized_image = resize(cropped_image, self.crop_size)
        resized_label = resize(cropped_label, self.crop_size, is_label=True)
        case['image'] = resized_image
        case['label'] = resized_label

        return case

# def resize(input,
#            output_shape,
#            is_label=False,
#            order=1,
#            mode='reflect',
#            cval=0,
#            anti_aliasing=True):
#     '''
#     A wrap of scikit-image resize for label encoding data support.

#     Args:
#         See scikit-image doc resize for more detail.
#         is_label: If true, split label before resize.
#     '''
#     dtype = input.dtype
#     orig_shape = input.shape
#     assert len(output_shape) == len(orig_shape) or len(output_shape) == len(orig_shape)-1, \
#         'output shape not equal to input shape'
#     if is_label:
#         num_classes = len(np.unique(input))

#     if order == 0 or not is_label or num_classes < 3:
#         if len(output_shape) == len(orig_shape)-1:
#             resized_input = np.array([tf.resize(c.astype(np.float32),
#                                                 output_shape,
#                                                 order,
#                                                 mode=mode,
#                                                 cval=cval,
#                                                 anti_aliasing=anti_aliasing)
#                                       for c in to_tensor(input)])
#             return to_numpy(resized_input)
#         else:
#             return tf.resize(input.astype(np.float32),
#                              output_shape,
#                              order,
#                              mode=mode,
#                              cval=cval,
#                              anti_aliasing=anti_aliasing).astype(dtype)
#     else:
#         num_classes = len(np.unique(input))
#         onehot = to_one_hot(input, num_classes, to_tensor=True)
#         resized_onehot = np.array([tf.resize(c.astype(np.float32),
#                                              output_shape,
#                                              order,
#                                              mode=mode,
#                                              cval=cval,
#                                              anti_aliasing=anti_aliasing)
#                                    for c in onehot])

#         return np.argmax(resized_onehot, axis=0).astype(dtype)


# def rescale(input,
#             scale,
#             is_label=False,
#             order=1,
#             mode='reflect',
#             cval=0,
#             multichannel=False,
#             anti_aliasing=True):
#     '''
#     A wrap of scikit-image rescale for label encoding data support.

#     Args:
#         See scikit-image doc rescale for more detail.
#         is_label: If true, split label before rescale.
#     '''
#     dtype = input.dtype

#     if is_label:
#         num_classes = len(np.unique(input))

#     if order == 0 or not is_label or num_classes < 3:
#         # why not using multichannel arg in tf.rescale? because it takes more memoey.
#         if multichannel:
#             rescaled = np.array([tf.rescale(c.astype(np.float32),
#                                             scale,
#                                             order,
#                                             mode=mode,
#                                             cval=cval,
#                                             anti_aliasing=anti_aliasing)
#                                  for c in to_tensor(input)])
#             return to_numpy(rescaled)
#         else:
#             return tf.rescale(input.astype(np.float32),
#                               scale,
#                               order,
#                               mode=mode,
#                               cval=cval,
#                               anti_aliasing=anti_aliasing).astype(dtype)
#     else:
#         num_classes = len(np.unique(input))
#         onehot = to_one_hot(input, num_classes, to_tensor=True)
#         rescale_onehot = np.array([tf.rescale(c.astype(np.float32),
#                                               scale,
#                                               order,
#                                               mode=mode,
#                                               cval=cval,
#                                               anti_aliasing=anti_aliasing)
#                                    for c in onehot])
#         return np.argmax(rescale_onehot, axis=0).astype(dtype)
