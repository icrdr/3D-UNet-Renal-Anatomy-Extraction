# %%
from data import batch_load_crop_case, analyze_cases, analyze_raw_cases,\
    batch_resample_normalize_case, batch_regions_crop_case, CaseDataset
from visualize import case_plt

image_dir = '/mnt/main/dataset/Task00_Kidney/imagesTr'
label_dir = '/mnt/main/dataset/Task00_Kidney/labelsTr'
crop_dir = 'data/Task00_Kidney/crop'
norm_dir = 'data/Task00_Kidney/norm'
region_crop_dir = 'data/Task00_Kidney/region_crop2'
region_norm_dir = 'data/Task00_Kidney/region_norm2'

normalize_stats = {
    "mean": 100.23331451416016,
    "std": 76.66192626953125,
    "pct_00_5": -79.0,
    "pct_99_5": 303.0
}
# %%

batch_load_crop_case(image_dir, crop_dir, -200)


# %%
batch_resample_normalize_case(crop_dir,
                              norm_dir,
                              (2.4, 2.4, 3),
                              normalize_stats)

# %%
batch_regions_crop_case(crop_dir, region_crop_dir, threshold=10000)

# %%

batch_resample_normalize_case(region_crop_dir,
                              region_norm_dir,
                              (0.78125, 0.78125, 1),
                              normalize_stats)

# %%
analyze_cases(crop_dir)

# %%
analyze_raw_cases(image_dir,label_dir)

# %%
