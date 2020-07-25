# %%
from data import batch_load_crop_case, analyze_cases, \
    batch_resample_normalize_case, batch_regions_crop_case, CaseDataset
from visualize import case_plt

from pathlib import Path
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import csv

image_dir = '/mnt/main/dataset/Task20_Kidney/imagesTr'
vessel_label_dir = '/mnt/main/dataset/Task20_Kidney/labelsTr_vessel'
vessel_crop_dir = 'data/Task20_Kidney/vessel_crop'
kidney_region_crop_dir = 'data/Task20_Kidney/kidney_region_crop'
vessel_region_crop_dir = 'data/Task20_Kidney/vessel_region_crop'
vessel_region_norm_dir = 'data/Task20_Kidney/vessel_region_norm'

region_based_on_dir = 'data/Task20_Kidney/kidney_label'

normalize_stats = {'median': 136.0,
                   'mean': 137.51925659179688,
                   'std': 88.92479705810547,
                   'min': -148.0,
                   'max': 1262.0,
                   'pct_00_5': -69.0,
                   'pct_99_5': 426.0}

# %%

batch_load_crop_case(image_dir,
                     vessel_label_dir,
                     vessel_crop_dir)

# %%
batch_regions_crop_case(vessel_crop_dir,
                        vessel_region_crop_dir,
                        threshold=10000,
                        pred_dir=region_based_on_dir)

# %%
batch_resample_normalize_case(vessel_region_crop_dir,
                              vessel_region_norm_dir,
                              (0.7, 0.7, 1),
                              normalize_stats)

# %%
analyze_cases(vessel_region_crop_dir)

# %%


def analyze_labels(load_dir, save_dir='chart/', num_labels=3, data_range=None):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    load_dir = Path(load_dir)
    cases = CaseDataset(load_dir)

    if data_range is None:
        data_range = range(len(cases))

    filename = "frequencies-{}.csv".format(datetime.now().strftime("%y%m%d%H%M"))
    csv_file = save_dir / filename
    with open(csv_file, 'w', newline='', encoding="utf-8-sig") as csvfile:
        csvWriter = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        for i in tqdm(data_range):
            case = cases[i]
            content = []
            for c in range(num_labels):
                label_np = case['label'] == c
                content.append(np.sum(label_np).item())

            csvWriter.writerow(content)


def analyze_annotations(load_dir, num_labels=4, data_range=None):
    load_dir = Path(load_dir)
    cases = CaseDataset(load_dir)
    sums = [0]*num_labels
    frequencies = [0]*num_labels
    if data_range is None:
        data_range = range(len(cases))

    for i in tqdm(data_range):
        case = cases[i]
        label_indices = np.unique(case['label'])
        for index in label_indices:
            sums[index] += 1
    frequencies = [v/len(data_range) for v in sums]
    print(sums)
    print(frequencies)
    return frequencies
    # plt.bar(sums)


analyze_labels(vessel_region_crop_dir, num_labels=3)


# %%
