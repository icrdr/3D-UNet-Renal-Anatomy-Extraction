
# %%
from torchvision.transforms import Compose
from transform import Crop, resize, rescale, to_one_hot, RandomCrop, ToOnehot, ToNumpy, ToTensor, \
    CombineLabels, RandomBrightness, RandomContrast, RandomGamma, RandomRescale, RandomRescaleCrop, \
    RandomMirror, split_dim, to_tensor, to_numpy
from data import CaseDataset, apply_scale, get_spacing
import numpy as np
from visualize import case_plt
import matplotlib.pyplot as plt
from skimage import measure
import scipy.ndimage as ndi
# %%


def remove_small_region(input, threshold):
    labels = measure.label(input)
    label_areas = np.bincount(labels.ravel())
    too_small_labels = label_areas < threshold
    too_small_mask = too_small_labels[labels]
    input[too_small_mask] = 0
    return input


patch_size = (160, 160, 80)

composed = Compose([
    # RandomRescale([1.2, 1.5]),
    # RandomMirror((0.9, 0, 0)),
    # RandomContrast(0.6),
    # RandomBrightness(0.6),
    # RandomGamma(0.6),
    # CombineLabels([0, 2], 3),
    # Crop(patch_size, enforce_label_indices=[1, 2], crop_mode='random'),
    RemoveSmallRegion(1000),
])

dataset = CaseDataset('data/Task03_Liver/normalized', transform=composed)
sample = dataset[2]
image = sample['image']
label = sample['label']

# %%

case_plt(casev, slice_pct=0.4, axi=2)


# %%
plt.imshow(label[:, :, 50], vmin=0, vmax=2)


# %%
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
        num_classes = len(np.unique(input))

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


def resample_normalize_case(case, target_spacing, normalize_stats):
    '''
    Reasmple image and label to target spacing, than normalize the image ndarray.

    Args:
        image: Image ndarray.
        meta: Meta info of this image ndarray.
        target_spacing: Target spacing for resample.
        normalize_stats: Intesity statstics dict used for normalize.
            {
                'mean':
                'std':
                'pct_00_5':
                'pct_99_5':
            }

    Return:
        image: cropped image ndarray
        label: cropped label ndarray
        meta: add more meta info to the dict:
            resampled_shape: Recased ndarray shape.
            resampled_spacing: Recased ndarray spacing.

    Example:
        case = resample_normalize_case(case,
                                    target_spacing=(1,1,3),
                                    normalize_stats={
                                        'mean':100,
                                        'std':50,
                                        'pct_00_5':-1024
                                        'pct_99_5':1024
                                    })
    '''
    case = case.copy()
    if not isinstance(normalize_stats, list):
        normalize_stats = [normalize_stats]

    # resample
    scale = (np.array(get_spacing(case['affine'])) / np.array(target_spacing))
    image_arr = rescale(case['image'], scale, multi_class=True)

    # normalize
    image_arr_list = []
    image_per_c = split_dim(image_arr)

    for c, s in enumerate(normalize_stats):
        mean = s['mean']
        std = s['std']
        pct_00_5 = s['pct_00_5']
        pct_99_5 = s['pct_99_5']

        cliped = np.clip(image_per_c[c], pct_00_5, pct_99_5)
        image_arr_list.append((cliped-mean)/(std+1e-8))

    case['image'] = np.stack(image_arr_list, axis=-1)

    if 'label' in case:
        case['label'] = rescale(case['label'], scale, is_label=True)

    case['affine'] = apply_scale(case['affine'], 1 / scale)

    return case


cases = CaseDataset('data/Task00_Kidney/region_crop')
case = cases[18]
print(case['case_id'])

# %%
target_spacing = (0.78125, 0.78125, 1)
casev = case.copy()
scale = (np.array(get_spacing(case['affine'])) / np.array(target_spacing))
casev['label'] = rescale(casev['label'], scale, is_label=True)

# %%
