
from transform import rescale, split_dim, crop_pad_to_bbox,\
    combination_labels, remove_small_region
import torch
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from utils import json_load, json_save
import numpy as np
import scipy.ndimage as ndi
from transforms3d.affines import compose, decompose


class CaseDataset(torch.utils.data.Dataset):
    '''
    A dataset class for loading preprocessed data.

    Args:
        load_dir: MSD (Medical Segmentation Decathlon) like task folder path.
        transform: list of transforms or composed transforms.
        load_meta: load meta info of the case or not.
    Example:
        cases = CaseDataset('/Task00_KD')
        case = cases[0]
    '''

    def __init__(self, load_dir, transform=None):
        super(CaseDataset, self).__init__()
        self.load_dir = Path(load_dir)
        self.transform = transform
        self.image_files = sorted(list(self.load_dir.glob('*.image.nii.gz')))
        self.label_files = sorted(list(self.load_dir.glob('*.label.nii.gz')))
        if len(self.image_files) == len(self.label_files):
            self.load_label = True

    def __getitem__(self, index):
        image_nib = nib.load(str(self.image_files[index]))
        case = {'case_id': str(self.image_files[index]).split('/')[-1].split('.')[0],
                'affine': image_nib.affine,
                'image': image_nib.get_fdata().astype(np.float32)}

        if self.load_label:
            label_nib = nib.load(str(self.label_files[index]))
            case['label'] = label_nib.get_fdata().astype(np.int64)

        if self.transform:
            case = self.transform(case)

        return case

    def __len__(self):
        return len(self.image_files)


def get_spacing(affine):
    spacing_x = np.linalg.norm(affine[0, :3])
    spacing_y = np.linalg.norm(affine[1, :3])
    spacing_z = np.linalg.norm(affine[2, :3])
    return (spacing_x, spacing_y, spacing_z)


def apply_scale(affine, scale):
    T, R, Z, S = decompose(affine)
    Z = Z * np.array(scale)
    return compose(T, R, Z, S)


def apply_translate(affine, offset):
    T, R, Z, S = decompose(affine)
    T = T + np.array(offset)
    return compose(T, R, Z, S)


def load_case(image_file, label_file=None):
    image_nib = nib.load(str(image_file))
    case = {'case_id': str(image_file).split('/')[-1].split('.')[0],
            'affine': image_nib.affine,
            'image': image_nib.get_fdata().astype(np.float32)}

    if label_file:
        label_nib = nib.load(str(label_file))
        case['label'] = label_nib.get_fdata().astype(np.int64)

    return case


def save_case(case, save_dir):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    image_fname = '%s.image.nii.gz' % case['case_id']
    image_nib = nib.Nifti1Pair(case['image'].astype(np.float32), case['affine'])
    nib.save(image_nib, str(save_dir / image_fname))

    if 'label' in case:
        label_fname = '%s.label.nii.gz' % case['case_id']
        label_nib = nib.Nifti1Pair(case['label'].astype(np.uint8), case['affine'])
        nib.save(label_nib, str(save_dir / label_fname))

    if 'pred' in case:
        pred_fname = '%s.pred.nii.gz' % case['case_id']
        pred_nib = nib.Nifti1Pair(case['pred'].astype(np.uint8), case['affine'])
        nib.save(pred_nib, str(save_dir / pred_fname))


def save_pred(case, save_dir):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    pred_fname = '%s.pred.nii.gz' % case['case_id']
    pred_nib = nib.Nifti1Pair(case['pred'].astype(np.uint8), case['affine'])
    nib.save(pred_nib, str(save_dir / pred_fname))


def orient_crop_case(case, air=-200):
    '''
    Load data file, orient to RAS sys, than crop to non-air bbox.

    Args:
        image_file: Image file path.
        label_file: Label file path.
        air: Air value to crop with. Any voxel value below this value, regard as air aera.

    Return:
        image: cropped image ndarray
        label: cropped label ndarray
        meta: meata info dict:
            affine: orient ndarray affine (matrix).
            spacing: orient ndarray spacing.
            shape: orient ndarray shape.
            shape: cropped ndarray RAS coord sys
    Example:
        case = load_crop_case('/Task00_KD/imagesTr/case_0001.nii',
                              '/Task00_KD/labelsTr/case_0001.nii',
                              -200)
    '''
    case = case.copy()
    orient = nib.orientations.io_orientation(case['affine'])
    image_nib = nib.Nifti1Pair(case['image'], case['affine'])
    image_nib = image_nib.as_reoriented(orient)
    image_arr = image_nib.get_fdata().astype(np.float32)

    if 'label' in case:
        label_nib = nib.Nifti1Pair(case['label'], case['affine'])
        label_nib = label_nib.as_reoriented(orient)
        label_arr = label_nib.get_fdata().astype(np.int64)

    if len(image_arr.shape) == 3:
        image_arr = np.expand_dims(image_arr, -1)

    # clac non-air box shape in all channel except label
    nonair_pos = [np.array(np.where(image > air)) for image in split_dim(image_arr)]
    nonair_min = np.array([nonair_pos.min(axis=1) for nonair_pos in nonair_pos])
    nonair_max = np.array([nonair_pos.max(axis=1) for nonair_pos in nonair_pos])
    # nonair_bbox shape (2,3) => (3,2)
    nonair_bbox = np.array([nonair_min.min(axis=0), nonair_max.max(axis=0)]).T
    nonair_bbox_ = np.concatenate([nonair_bbox, [[0, image_arr.shape[-1]]]])

    # cropping
    case['image'] = crop_pad_to_bbox(image_arr, nonair_bbox_)
    case['bbox'] = nonair_bbox

    if 'label' in case:
        case['label'] = crop_pad_to_bbox(label_arr, nonair_bbox)

    offset = nonair_bbox[:, 0] * get_spacing(image_nib.affine)
    case['affine'] = apply_translate(image_nib.affine, offset)

    return case


def batch_load_crop_case(image_dir, label_dir, save_dir, air=-200, data_range=None):
    '''
    Batch orient to RAS, crop to non-air bbox and than save as new case file.

    A case has a data npz file and a meta json file.
    [case_id]_data.npz contraines dnarray dict ['image'] (['label'])
    [case_id]_meta.json contraines all meta info of the case:
        case_id: The uid of the case, used for naming.
        orient: loaded ndarray coord sys to RAS coord sys.
        origial_coord_sys: loaded ndarray coord sys.
        origial_affine: loaded ndarray affine (matrix), ndarray space to physical space (RAS).
        origial_spacing: loaded ndarray spacing.
        origial_shape: loaded ndarray shape.
        coord_sys: cropped ndarray coord sys (RAS).
        affine: orient ndarray affine (matrix).
        spacing: orient ndarray spacing.
        shape: orient ndarray shape.
        cropped_shape: cropped ndarray RAS coord sys
        nonair_bbox: non-air boundary box:[[x_min,x_max],[y_min,y_max],[z_min,z_max]]

    props.json including info below:
        air: Setted air value.
        modality: modality name in MSD's dataset.json
        labels: labels name in MSD's dataset.json

    Args:
        load_dir: MSD (Medical Segmentation Decathlon) like task folder path.
        save_dir: Cropped data save folder path.
        porps_save_dir: Porps file save path.
        air: Air value to crop with. Any voxel value below this value, regard as air aera
        data_range: If u only want to load part of the case in the folder.
    '''
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_files = [path for path in sorted(image_dir.iterdir()) if path.is_file()]
    label_files = [path for path in sorted(label_dir.iterdir()) if path.is_file()]
    assert len(image_files) == len(label_files),\
        'number of images is not equal to number of labels.'

    if data_range is None:
        data_range = range(len(image_files))

    for i in tqdm(data_range):
        case = load_case(image_files[i], label_files[i])
        case = orient_crop_case(case, air)
        save_case(case, save_dir)


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


def batch_resample_normalize_case(load_dir,
                                  save_dir,
                                  target_spacing,
                                  normalize_stats,
                                  data_range=None):
    '''
    Batch resample & normalize, than saved as new case file.
    Adding following info to props file:
        resampled_spacing: target spacing setting by arg: spacing
        median_resampled_shape: median of shape after resample.
        normalize_statstics: all statstics used for normalize (see below).

    Args:
        load_dir: Data loaded from folder path.
        save_dir: Propressed data save folder path.
        target_spacing: Target spacing for resample.
        normalize_stats: Intesity statstics dict used for normalize.
            {
                'mean':
                'std':
                'pct_00_5':
                'pct_99_5':
            }
    '''
    load_dir = Path(load_dir)
    cases = CaseDataset(load_dir)

    if data_range is None:
        data_range = range(len(cases))

    for i in tqdm(data_range):
        case = resample_normalize_case(cases[i], target_spacing, normalize_stats)
        save_case(case, save_dir)


def analyze_cases(load_dir, props_file=None, data_range=None):
    '''
    Analyze all data in folder, calcate the modality statstics,
    median of spacing and median of shape. than add these info into props file.
    A modality statstics including below:
        median: Median of data intesity in the modality (class).
        mean: Mean of intesity in the modality (class).
        std: Standard deviation of intesity in the modality (class).
        min: Minimum of intesity in the modality (class).
        max: Maximum of intesity in the modality (class).
        pct_00_5: Percentile 00.5 of intesity in the modality (class).
        pct_99_5: Percentile 99.5 of intesity in the modality (class).

    Args:
        load_dir: Data folder path.
        props_file: Porps file path.

    Return:
        props: New generated props with original props from props file including:
            median_spacing: median of spacing.
            median_cropped_shape: median of shape.
            modality_statstics: all modality statstics (see above).
    '''
    load_dir = Path(load_dir)
    cases = CaseDataset(load_dir)

    shapes = []
    spacings = []
    n_modality = cases[0]['image'].shape[-1]
    modality_values = [[]*n_modality]

    if data_range is None:
        data_range = range(len(cases))

    for i in tqdm(data_range):
        case = cases[i]
        shapes.append(case['image'].shape[:-1])
        spacings.append(get_spacing(case['affine']))

        label_mask = np.array(case['label'] > 0)
        sub_images = split_dim(case['image'])
        for c in range(n_modality):
            voxels = sub_images[c][label_mask][::10]
            modality_values[c].append(voxels)

    modality_values = [np.concatenate(i) for i in modality_values]
    spacings = np.array(spacings)
    shapes = np.array(shapes)

    modality_statstics = []
    for c in range(n_modality):
        modality_statstics.append({
            'median': np.median(modality_values[c]).item(),
            'mean': np.mean(modality_values[c]).item(),
            'std': np.std(modality_values[c]).item(),
            'min': np.min(modality_values[c]).item(),
            'max': np.max(modality_values[c]).item(),
            'pct_00_5': np.percentile(modality_values[c], 00.5).item(),
            'pct_99_5': np.percentile(modality_values[c], 99.5).item()
        })

    new_props = {
        'max_spacing': np.max(spacings, axis=0).tolist(),
        'max_shape': np.max(shapes, axis=0).tolist(),
        'min_spacing': np.min(spacings, axis=0).tolist(),
        'min_shape': np.min(shapes, axis=0).tolist(),
        'mean_spacing': np.mean(spacings, axis=0).tolist(),
        'mean_shape': np.mean(shapes, axis=0).tolist(),
        'median_spacing': np.median(spacings, axis=0).tolist(),
        'median_shape': np.median(shapes, axis=0).tolist(),
        'modality_statstics': modality_statstics,
    }

    if props_file is not None:
        props_file = Path(props_file)
        props = json_load(str(props_file))
        props = {**props, **new_props}
        json_save(str(props_file), props)

    return new_props


def analyze_raw_cases(image_dir, label_dir, props_file=None, data_range=None):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_files = [path for path in sorted(image_dir.iterdir()) if path.is_file()]
    label_files = [path for path in sorted(label_dir.iterdir()) if path.is_file()]
    assert len(image_files) == len(label_files),\
        'number of images is not equal to number of labels.'

    shapes = []
    spacings = []
    modality_values = []

    if data_range is None:
        data_range = range(len(image_files))

    for i in tqdm(data_range):
        case = load_case(image_files[i], label_files[i])
        shapes.append(case['image'].shape)
        spacings.append(get_spacing(case['affine']))

        label_mask = np.array(case['label'] > 0)
        voxels = case['image'][label_mask][::10]
        modality_values.append(voxels)

    modality_values = np.concatenate(modality_values)
    spacings = np.array(spacings)
    shapes = np.array(shapes)

    modality_statstics = {
        'median': np.median(modality_values).item(),
        'mean': np.mean(modality_values).item(),
        'std': np.std(modality_values).item(),
        'min': np.min(modality_values).item(),
        'max': np.max(modality_values).item(),
        'pct_00_5': np.percentile(modality_values, 00.5).item(),
        'pct_99_5': np.percentile(modality_values, 99.5).item()
    }

    new_props = {
        'max_spacing': np.max(spacings, axis=0).tolist(),
        'max_shape': np.max(shapes, axis=0).tolist(),
        'min_spacing': np.min(spacings, axis=0).tolist(),
        'min_shape': np.min(shapes, axis=0).tolist(),
        'mean_spacing': np.mean(spacings, axis=0).tolist(),
        'mean_shape': np.mean(shapes, axis=0).tolist(),
        'median_spacing': np.median(spacings, axis=0).tolist(),
        'median_shape': np.median(shapes, axis=0).tolist(),
        'modality_statstics': modality_statstics,
    }

    if props_file is not None:
        props_file = Path(props_file)
        props = json_load(str(props_file))
        props = {**props, **new_props}
        json_save(str(props_file), props)

    return new_props


def regions_crop_case(case, threshold=0, padding=20, based_on='label'):
    if based_on == 'label':
        based = case['label'] > 0
    elif based_on == 'pred':
        based = case['pred'] > 0

    based = remove_small_region(based, threshold)
    labels, nb_labels = ndi.label(based)
    objects = ndi.find_objects(labels)
    regions = []
    padding = np.round(padding / np.array(get_spacing(case['affine']))).astype(np.int)
    for i, slices in enumerate(objects):
        region_bbox = np.array([[slices[0].start-padding[0], slices[0].stop+padding[0]],
                                [slices[1].start-padding[1], slices[1].stop+padding[1]],
                                [slices[2].start-padding[2], slices[2].stop+padding[2]]])
        region_bbox_ = np.concatenate([region_bbox, [[0, case['image'].shape[-1]]]])
        offset = region_bbox[:, 0] * get_spacing(case['affine'])
        # crop
        region = {'case_id': '%s_%03d' % (case['case_id'], i),
                  'affine': apply_translate(case['affine'], offset),
                  'bbox': region_bbox,
                  'image': crop_pad_to_bbox(case['image'], region_bbox_)}

        if 'label'in case:
            region['label'] = crop_pad_to_bbox(case['label'], region_bbox)

        regions.append(region)

    return regions


def batch_regions_crop_case(load_dir,
                            save_dir,
                            threshold=0,
                            padding=20,
                            pred_dir=None,
                            data_range=None):
    load_dir = Path(load_dir)
    cases = CaseDataset(load_dir)

    if pred_dir is not None:
        pred_dir = Path(pred_dir)
        pred_files = sorted(list(pred_dir.glob('*.pred.nii.gz')))

    if data_range is None:
        if pred_dir is not None:
            data_range = range(len(pred_files))
        else:
            data_range = range(len(cases))

    for i in tqdm(data_range):
        case = cases[i]
        if pred_dir is not None:
            based_on = 'pred'
            pred_nib = nib.load(str(pred_files[i]))
            case['pred'] = pred_nib.get_fdata().astype(np.int64)
        else:
            based_on = 'label'

        regions = regions_crop_case(case,
                                    threshold,
                                    padding,
                                    based_on)
        for region in regions:
            save_case(region, save_dir)
