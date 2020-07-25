# %%
from skimage.color import label2rgb
from tqdm import tqdm
import nibabel as nib
from pathlib import Path
from visualize import case_plt
from trainer import predict_case, cascade_predict_case, cascade_predict, evaluate_case, \
    batch_evaluate, batch_cascade_predict
from data import CaseDataset, save_pred, save_case
from network import ResUnet3D, ResAttrUnet3D, ResAttrBNUnet3D
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from transform import remove_small_region


ckpt1 = torch.load('logs/Task00_Kidney/kd-2004070628-epoch=54.pt')
ckpt2 = torch.load('logs/DOC/iia-H-09-last.pt')
coarse_model = ResAttrUnet3D(out_channels=1).cuda()
detail_model = ResUnet3D(out_channels=4).cuda()
coarse_model.load_state_dict(ckpt1['model_state_dict'])
detail_model.load_state_dict(ckpt2['model_state_dict'])

normalize_stats1 = {"mean": 100.23331451416016,
                    "std": 76.66192626953125,
                    "pct_00_5": -79.0,
                    "pct_99_5": 303.0}

normalize_stats2 = {'median': 125.0,
                    'mean': 118.8569564819336,
                    'std': 60.496273040771484,
                    'pct_00_5': -28.0,
                    'pct_99_5': 255.0}

# %%
cases = CaseDataset('data/Task00_Kidney/crop')
case = cascade_predict_case(cases[149],
                            coarse_model=coarse_model,
                            coarse_target_spacing=(2.4, 2.4, 3),
                            coarse_normalize_stats=normalize_stats1,
                            coarse_patch_size=(144, 144, 96),
                            detail_model=detail_model,
                            detail_target_spacing=(0.7, 0.7, 1),
                            detail_normalize_stats=normalize_stats2,
                            detail_patch_size=(128, 128, 128),
                            num_classes=4)

# %%
image_file = '/mnt/main/dataset/Task20_Kidney/imagesTs/0659212.nii.gz'
# image_file = '/mnt/main/dataset/Task20_Kidney/image2.nii.gz'
label_file = None
# image_file = '/mnt/main/dataset/Task00_Kidney/imagesTr/case_00005.nii.gz'
# label_file = '/mnt/main/dataset/Task00_Kidney/labelsTr/case_00005.nii.gz'

case = cascade_predict(image_file=image_file,
                       label_file=label_file,
                       coarse_model=coarse_model,
                       coarse_target_spacing=(2.4, 2.4, 3),
                       coarse_normalize_stats=normalize_stats1,
                       coarse_patch_size=(144, 144, 96),
                       detail_model=detail_model,
                       detail_target_spacing=(0.7, 0.7, 1),
                       detail_normalize_stats=normalize_stats2,
                       detail_patch_size=(128, 128, 128),
                       num_classes=4)
# %%
cases = CaseDataset('data/Task00_Kidney/region_crop')
case = predict_case(cases[197],
                    model=detail_model,
                    target_spacing=(1, 1, 1),
                    normalize_stats=normalize_stats2,
                    patch_size=(96, 96, 144),
                    num_classes=4)
# %%
evaluate_case(case)
# %%
case_plt(case, slice_pct=0.5, axi=2)
# %%
save_case(case, './')

# %%
image_dir = '/mnt/main/dataset/Task20_Kidney/imagesTr'
test_dir = '/mnt/main/dataset/Task20_Kidney/imagesTs'
pred_dir = '/mnt/main/dataset/Task20_Kidney/predictsTr_09_kidney'

batch_cascade_predict(image_dir,
                      pred_dir,
                      coarse_model=coarse_model,
                      coarse_target_spacing=(2.4, 2.4, 3),
                      coarse_normalize_stats=normalize_stats1,
                      coarse_patch_size=(144, 144, 96),
                      detail_model=detail_model,
                      detail_target_spacing=(0.7, 0.7, 1.0),
                      detail_normalize_stats=normalize_stats2,
                      detail_patch_size=(128, 128, 128),
                      num_classes=4)
# %%
case = CaseDataset('data/Task20_Kidney/region_norm_kidney')[10]

image = case['image'][:, :, 50, 0]/3+1
label_image = case['label'][:, :, 50]
image_label_overlay = label2rgb(label_image, image=image)
plt.imshow(image_label_overlay)

# %%
print(case['pred'].max())

# %%
