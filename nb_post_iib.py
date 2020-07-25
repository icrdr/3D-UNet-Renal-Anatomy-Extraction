# %%
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
ckpt2 = torch.load('logs/DOC/iib-H-09-last.pt')
coarse_model = ResAttrUnet3D(out_channels=1).cuda()
detail_model = ResUnet3D(out_channels=3).cuda()
coarse_model.load_state_dict(ckpt1['model_state_dict'])
detail_model.load_state_dict(ckpt2['model_state_dict'])

normalize_stats1 = {"mean": 100.23331451416016,
                    "std": 76.66192626953125,
                    "pct_00_5": -79.0,
                    "pct_99_5": 303.0}


normalize_stats2 = {'median': 136.0,
                   'mean': 137.51925659179688,
                   'std': 88.92479705810547,
                   'pct_00_5': -69.0,
                   'pct_99_5': 426.0}

# %%
cases = CaseDataset('data/Task00_Kidney/region_crop')

case = predict_case(cases[5],
                    model=detail_model,
                    target_spacing=(0.7, 0.7, 1),
                    normalize_stats=normalize_stats2,
                    patch_size=(128, 128, 128))

# %%
image_file = '/mnt/main/dataset/Task20_Kidney/imagesTr/case_010.nii.gz'
label_file = None
# image_file = '/mnt/main/dataset/Task00_Kidney/imagesTr/case_00005.nii.gz'
label_file = '/mnt/main/dataset/Task20_Kidney/labelsTr_vessel/case_010.nii.gz'

case = cascade_predict(image_file=image_file,
                       label_file=label_file,
                       coarse_model=coarse_model,
                       coarse_target_spacing=(2.4, 2.4, 3),
                       coarse_normalize_stats=normalize_stats1,
                       coarse_patch_size=(144, 144, 96),
                       detail_model=detail_model,
                       detail_target_spacing=(0.7, 0.7, 1),
                       detail_normalize_stats=normalize_stats2,
                       detail_patch_size=(128, 128, 128))
# %%
evaluate_case(case)
# %%
case_plt(case, slice_pct=0.4, axi=2)
# %%
save_case(case, './')


# %%
# %%
image_dir = '/mnt/main/dataset/Task20_Kidney/imagesTr'
test_dir = '/mnt/main/dataset/Task20_Kidney/imagesTs'
pred_dir = '/mnt/main/dataset/Task20_Kidney/predictsTr_09_vessel'

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
                      num_classes=3)

# %%
