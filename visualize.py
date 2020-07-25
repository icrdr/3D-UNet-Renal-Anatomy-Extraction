
import matplotlib.pyplot as plt
import numpy as np


def grid_plt(grid_list, value_ranges=None):
    rows = len(grid_list)
    cols = len(grid_list[0])
    plt.figure(figsize=(cols*4, rows*4))
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, cols*i + j + 1)
            if value_ranges:
                plt.imshow(grid_list[i][j],
                           vmin=value_ranges[j][0],
                           vmax=value_ranges[j][1])
            else:
                plt.imshow(grid_list[i][j])
            plt.axis('off')
    plt.tight_layout()
    plt.show()


def case_plt(case, slice_pct=0.5, axi=0, one_hot_label=False, one_hot_pred=False):
    image = case['image']
    label = case['label'] if'label' in case else None
    pred = case['pred'] if'pred' in case else None

    slice_index = round(image.shape[axi]*slice_pct)

    grid_list = []
    value_ranges = []
    for c in range(image.shape[-1]):
        if axi == 0:
            grid_list.append(image[slice_index, :, :, c])
        elif axi == 1:
            grid_list.append(image[:, slice_index, :, c])
        else:
            grid_list.append(image[:, :, slice_index, c])
        value_ranges.append([np.percentile(image, 00.5),
                             np.percentile(image, 99.5)])

    if label is not None:
        if one_hot_label:
            for c in range(label.shape[-1]):
                if axi == 0:
                    grid_list.append(label[slice_index, :, :, c])
                elif axi == 1:
                    grid_list.append(label[:, slice_index, :, c])
                else:
                    grid_list.append(label[:, :, slice_index, c])
                value_ranges.append([0, 1])
        else:
            if axi == 0:
                grid_list.append(label[slice_index, :, :])
            elif axi == 1:
                grid_list.append(label[:, slice_index, :])
            else:
                grid_list.append(label[:, :, slice_index])
            value_ranges.append([label.min(), label.max()])

    if pred is not None:
        if one_hot_pred:
            for c in range(label.shape[-1]):
                if axi == 0:
                    grid_list.append(pred[slice_index, :, :, c])
                elif axi == 1:
                    grid_list.append(pred[:, slice_index, :, c])
                else:
                    grid_list.append(pred[:, :, slice_index, c])
                value_ranges.append([0, 1])
        else:
            if axi == 0:
                grid_list.append(pred[slice_index, :, :])
            elif axi == 1:
                grid_list.append(pred[:, slice_index, :])
            else:
                grid_list.append(pred[:, :, slice_index])
            value_ranges.append([pred.min(), pred.max()])

    grid_plt([grid_list], value_ranges=value_ranges)
