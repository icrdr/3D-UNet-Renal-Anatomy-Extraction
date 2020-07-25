# %%
import shutil
from tqdm import tqdm
import nibabel as nib
from pathlib import Path
from visualize import case_plt
from trainer import cascade_predict_case, cascade_predict, evaluate_case, \
    batch_evaluate, batch_cascade_predict
from data import CaseDataset, save_pred
from network import ResUnet3D
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from transform import remove_small_region

# %%
ckpt1 = torch.load('logs/Task00_Kidney/kd-2004070628-epoch=54.pt')
ckpt2 = torch.load('logs/Task00_Kidney/ca-2004080007-epoch=312.pt')
coarse_model = ResUnet3D(out_channels=1).cuda()
detail_model = ResUnet3D(out_channels=3).cuda()
coarse_model.load_state_dict(ckpt1['model_state_dict'])
detail_model.load_state_dict(ckpt2['model_state_dict'])

normalize_stats = {
    "mean": 100.23331451416016,
    "std": 76.66192626953125,
    "pct_00_5": -79.0,
    "pct_99_5": 303.0
}
cases = CaseDataset('data/Task00_Kidney/crop')

# %%
case = cascade_predict_case(cases[82],
                            coarse_model=coarse_model,
                            coarse_target_spacing=(2.4, 2.4, 3),
                            coarse_normalize_stats=normalize_stats,
                            coarse_patch_size=(144, 144, 96),
                            detail_model=detail_model,
                            detail_target_spacing=(0.78125, 0.78125, 1),
                            detail_normalize_stats=normalize_stats,
                            detail_patch_size=(96, 96, 144))
# %%
image_file = '/mnt/main/dataset/Task00_Kidney/imagesTs/case_00210.nii.gz'
label_file = None
# image_file = '/mnt/main/dataset/Task00_Kidney/imagesTr/case_00005.nii.gz'
# label_file = '/mnt/main/dataset/Task00_Kidney/labelsTr/case_00005.nii.gz'


case = cascade_predict(image_file=image_file,
                       label_file=label_file,
                       coarse_model=coarse_model,
                       coarse_target_spacing=(2.4, 2.4, 3),
                       coarse_normalize_stats=normalize_stats,
                       coarse_patch_size=(144, 144, 96),
                       detail_model=detail_model,
                       detail_target_spacing=(0.78125, 0.78125, 1),
                       detail_normalize_stats=normalize_stats,
                       detail_patch_size=(128, 128, 128))


# %%
evaluate_case(case)
# %%
case_plt(case, slice_pct=0.3, axi=0)
# %%
save_pred(case, './')
# %%
image_dir = '/mnt/main/dataset/Task00_Kidney/imagesTr'
test_dir = '/mnt/main/dataset/Task00_Kidney/imagesTs'
label_dir = '/mnt/main/dataset/Task00_Kidney/labelsTr'
pred_dir = '/mnt/main/dataset/Task00_Kidney/aaa'

batch_cascade_predict(test_dir,
                      pred_dir,
                      coarse_model=coarse_model,
                      coarse_target_spacing=(2.4, 2.4, 3),
                      coarse_normalize_stats=normalize_stats,
                      coarse_patch_size=(144, 144, 96),
                      detail_model=detail_model,
                      detail_target_spacing=(0.78125, 0.78125, 1),
                      detail_normalize_stats=normalize_stats,
                      detail_patch_size=(128, 128, 128),
                      data_range=None)


# %%


def create_sphere(shape, center, r):
    coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])
                       ** 2 + (coords[2]-center[2])**2)
    return 1*(distance <= r)


def post_transform(input, r=2):
    output = np.zeros_like(input)
    structure = create_sphere((7, 7, 7), (3, 3, 3), 4)

    mask = input > 0
    mask = remove_small_region(mask, 10000)
    # mask = ndi.binary_closing(mask)
    # mask = ndi.binary_opening(mask)
    output[mask] = 1

    kd = input == 2
    kd = ndi.binary_closing(kd, structure)
    kd = ndi.binary_opening(kd)
    output[kd] = 2

    return output


def batch_post_transform(load_dir, save_dir, data_range=None):
    load_dir = Path(load_dir)
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    pred_files = sorted(list(load_dir.glob('*.nii.gz')))

    if data_range is None:
        data_range = range(len(pred_files))

    for i in tqdm(data_range):
        pred_nib = nib.load(str(pred_files[i]))
        pred_arr = pred_nib.get_fdata().astype(np.uint8)
        affine = pred_nib.affine
        case_id = str(pred_files[i]).split('/')[-1].split('.')[0]

        pred_arr = post_transform(pred_arr)

        pred_fname = '%s.pred.nii.gz' % case_id
        pred_nib = nib.Nifti1Pair(pred_arr, affine)
        nib.save(pred_nib, str(save_dir / pred_fname))


# %%
casev = case.copy()
casev['pred'] = post_transform(casev['pred'])

# %%
pred_dir = '/mnt/main/dataset/Task00_Kidney/bbb'
pred_dir2 = '/mnt/main/dataset/Task00_Kidney/bbb2'
batch_post_transform(pred_dir, pred_dir2)

# %%
label_dir = '/mnt/main/dataset/Task00_Kidney/labelsTr'
pred_dir = '/mnt/main/dataset/Task00_Kidney/predictionsTr2'

batch_evaluate(label_dir, pred_dir, data_range=range(90))

# %%
print(ckpt2['current_epoch'])

# %%
load_dir = Path('/mnt/main/dataset/Task20_Kidney/kidney_labelsTr')
save_dir = Path('/mnt/main/dataset/Task20_Kidney/kidney_labelsTr_')

if not save_dir.exists():
    save_dir.mkdir(parents=True)

pred_files = sorted(list(load_dir.glob('*.nii.gz')))

for i in tqdm(range(len(pred_files))):
    pred_nib = nib.load(str(pred_files[i]))
    pred_arr = pred_nib.get_fdata().astype(np.uint8)
    output = np.zeros_like(pred_arr)
    mask = pred_arr > 0
    cacy = pred_arr > 1
    ca = pred_arr == 2
    mask = ndi.binary_erosion(mask)
    cacy = ndi.binary_erosion(cacy)
    ca = ndi.binary_erosion(ca)
    output[mask] = 1
    output[cacy] = 3
    output[ca] = 2

    affine = pred_nib.affine
    f_name = str(pred_files[i]).split('/')[-1]
    pred_nib = nib.Nifti1Pair(output, affine)
    nib.save(pred_nib, str(save_dir / f_name))


# %%
load_dir = Path('/mnt/main/ok')
image_dir = Path('/mnt/main/dataset/Task20_Kidney/imagesTr')
kidney_labels_dir = Path('/mnt/main/dataset/Task20_Kidney/labelsTr_kidney')
vessel_labels_dir = Path('/mnt/main/dataset/Task20_Kidney/labelsTr_vessel')

image_dir.mkdir(parents=True)
kidney_labels_dir.mkdir(parents=True)
vessel_labels_dir.mkdir(parents=True)

case_dirs = [path for path in sorted(load_dir.iterdir()) if path.is_dir()]

for i, case_dir in tqdm(enumerate(case_dirs)):
    case_id = "case_%03d.nii.gz" % i
    shutil.copy(str(case_dir / 'image.nii.gz'), str(image_dir / case_id))
    shutil.copy(str(case_dir / 'kidney_label.nii.gz'), str(kidney_labels_dir / case_id))
    shutil.copy(str(case_dir / 'vessel_label.nii.gz'), str(vessel_labels_dir / case_id))


# %%
# %%
load_dir = Path('/mnt/main/dataset/Task20_Kidney/predictsTr_09_vessel')
save_dir = Path('/mnt/main/dataset/Task20_Kidney/predictsTr_09_vessel_')

if not save_dir.exists():
    save_dir.mkdir(parents=True)

pred_files = sorted(list(load_dir.glob('*.nii.gz')))

for i in tqdm(range(len(pred_files))):
    pred_nib = nib.load(str(pred_files[i]))
    pred_arr = pred_nib.get_fdata().astype(np.uint8)
    output = np.zeros_like(pred_arr)
    mask = pred_arr > 0
    ar = pred_arr == 1
    ve = pred_arr == 2
    ar = ndi.binary_erosion(ar)
    ve = ndi.binary_erosion(ve)
    output[ar] = 1
    output[ve] = 2

    affine = pred_nib.affine
    f_name = str(pred_files[i]).split('/')[-1]
    pred_nib = nib.Nifti1Pair(output, affine)
    nib.save(pred_nib, str(save_dir / f_name))


# %%
pred_files = Path('/mnt/main/dataset/Task20_Kidney/predictsTr_05_vessel/case_015.pred.nii.gz')
save_dir = Path('/mnt/main/dataset/Task20_Kidney/predictsTr_05_vessel_')
pred_nib = nib.load(str(pred_files))
pred_arr = pred_nib.get_fdata().astype(np.uint8)
output = np.zeros_like(pred_arr)
el = ndi.generate_binary_structure(3, 2)
ar = pred_arr == 1
ve = pred_arr == 2

ar = ndi.binary_erosion(ar)
# ar = ndi.binary_opening(ar)
ar = ndi.binary_dilation(ar)
# ar = ndi.binary_closing(ar)
ve = ndi.binary_erosion(ve)
# ve = ndi.binary_opening(ve)
ve = ndi.binary_dilation(ve)
# ve = ndi.binary_closing(ve)

output[ar] = 1
output[ve] = 2

affine = pred_nib.affine
f_name = str(pred_files).split('/')[-1]
pred_nib = nib.Nifti1Pair(output, affine)
nib.save(pred_nib, str(save_dir / f_name))


# %%
