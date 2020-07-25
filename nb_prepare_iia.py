# %%
from data import batch_load_crop_case, analyze_cases, analyze_raw_cases,\
    batch_resample_normalize_case, batch_regions_crop_case, CaseDataset
from visualize import case_plt

image_dir = '/mnt/main/dataset/Task20_Kidney/imagesTr'
kidney_label_dir = '/mnt/main/dataset/Task20_Kidney/labelsTr_kidney'
kidney_crop_dir = 'data/Task20_Kidney/kidney_crop'
kidney_norm_dir = 'data/Task20_Kidney/kidney_norm'


kidney_region_crop_dir = 'data/Task20_Kidney/kidney_region_crop'
kidney_region_norm_dir = 'data/Task20_Kidney/kidney_region_norm'



normalize_stats = {'median': 125.0,
                   'mean': 118.8569564819336,
                   'std': 60.496273040771484,
                   'min': -189.0,
                   'max': 1075.0,
                   'pct_00_5': -28.0,
                   'pct_99_5': 255.0}

# %%

batch_load_crop_case(image_dir,
                     kidney_label_dir,
                     kidney_crop_dir)


# %%
batch_resample_normalize_case(kidney_crop_dir,
                              kidney_norm_dir,
                              (2.5, 2.5, 2),
                              normalize_stats)

# %%
batch_regions_crop_case(kidney_crop_dir,
                        kidney_region_crop_dir,
                        threshold=10000)

# %%
batch_resample_normalize_case(kidney_region_crop_dir,
                              kidney_region_norm_dir,
                              (0.7, 0.7, 1),
                              normalize_stats)

# %%
analyze_cases(kidney_crop_dir)


# %%
analyze_raw_cases(image_dir, kidney_label_dir)

# %%
