# %%
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import csv


def evaluate_metrics(input, target, smooth=1e-7):
    p = input.contiguous().view(-1)
    g = target.contiguous().view(-1)
    true_pos = (p * g).sum()
    true_neg = ((1-p) * (1-g)).sum()
    false_pos = (p * (1-g)).sum()
    false_neg = ((1-p) * g).sum()
    dsc = (true_pos + smooth)/(true_pos + 0.5*(false_neg + false_pos) + smooth)
    sen = (true_pos + smooth) / (true_pos + false_neg + smooth)
    spe = (true_neg + smooth) / (true_neg + false_pos + smooth)
    acc = (true_pos + true_neg + smooth) / (true_pos+true_neg+false_neg+false_pos+smooth)
    return {'dsc': dsc.item(),
            'sen': sen.item(),
            'spe': spe.item(),
            'acc': acc.item()}


def evaluate_case(case):
    num_classes = case['label'].max()
    evaluate_result = []
    for c in range(num_classes):
        pred = np.array(case['pred'] == c+1).astype(np.float32)
        label = np.array(case['label'] == c+1).astype(np.float32)
        metrics = evaluate_metrics(torch.tensor(pred), torch.tensor(label))
        evaluate_result.append(metrics)
    return evaluate_result


def evaluate(label_file, pred_file):

    label_nib = nib.load(str(label_file))
    pred_nib = nib.load(str(pred_file))

    case = {}
    case['label'] = label_nib.get_fdata().astype(np.uint8)
    case['pred'] = pred_nib.get_fdata().astype(np.uint8)
    evaluate_result = evaluate_case(case)
    return evaluate_result


def batch_evaluate(label_dir, pred_dir, save_dir='chart/', data_range=None):
    label_dir = Path(label_dir)
    pred_dir = Path(pred_dir)
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    label_files = sorted(list(label_dir.glob('*.nii.gz')))
    pred_files = sorted(list(pred_dir.glob('*.nii.gz')))

    if data_range is None:
        data_range = range(len(label_files))

    evaluate_results = []
    for i in tqdm(data_range):
        evaluate_result = evaluate(label_files[i], pred_files[i])
        evaluate_results.append(evaluate_result)

    time = datetime.now().strftime("%y%m%d%H%M")
    dsc_file = "%s-DSC.csv" % time
    csv_file = save_dir / dsc_file
    with open(csv_file, 'w', newline='', encoding="utf-8-sig") as csvfile:
        csvWriter = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        for evaluate_result in evaluate_results:
            content = []
            for result in evaluate_result:
                content.append(result['dsc'])

            csvWriter.writerow(content)

    sen_file = "%s-SEN.csv" % time
    csv_file = save_dir / sen_file
    with open(csv_file, 'w', newline='', encoding="utf-8-sig") as csvfile:
        csvWriter = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        for evaluate_result in evaluate_results:
            content = []
            for result in evaluate_result:
                content.append(result['sen'])

            csvWriter.writerow(content)

    spe_file = "%s-SPE.csv" % time
    csv_file = save_dir / spe_file
    with open(csv_file, 'w', newline='', encoding="utf-8-sig") as csvfile:
        csvWriter = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        for evaluate_result in evaluate_results:
            content = []
            for result in evaluate_result:
                content.append(result['spe'])

            csvWriter.writerow(content)

    acc_file = "%s-ACC.csv" % time
    csv_file = save_dir / acc_file
    with open(csv_file, 'w', newline='', encoding="utf-8-sig") as csvfile:
        csvWriter = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        for evaluate_result in evaluate_results:
            content = []
            for result in evaluate_result:
                content.append(result['acc'])

            csvWriter.writerow(content)


label_dir = '/mnt/main/dataset/Task20_Kidney/labelsTr_vessel/'
pred_dir = '/mnt/main/dataset/Task20_Kidney/predictsTr_09_vessel/'
batch_evaluate(label_dir, pred_dir)

# %%
