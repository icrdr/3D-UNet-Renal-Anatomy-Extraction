import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from loss import dice
from pathlib import Path
from data import CaseDataset, load_case, save_pred, \
    orient_crop_case, regions_crop_case, resample_normalize_case
import nibabel as nib
import numpy as np
import scipy.special as spe
from transform import pad, crop_pad, to_numpy, to_tensor, resize


def predict_per_patch(input,
                      model,
                      num_classes=3,
                      patch_size=(96, 96, 96),
                      step_per_patch=4,
                      verbose=True,
                      one_hot=False):
    device = next(model.parameters()).device
    # add padding if patch is larger than input shape
    origial_shape = input.shape[:3]
    input = pad(input, patch_size)
    padding_shape = input.shape[:3]
    coord_start = np.array([i // 2 for i in patch_size])
    coord_end = np.array([padding_shape[i] - patch_size[i] // 2
                          for i in range(len(patch_size))])
    num_steps = np.ceil([(coord_end[i] - coord_start[i]) / (patch_size[i] / step_per_patch)
                         for i in range(3)])
    step_size = np.array([(coord_end[i] - coord_start[i]) / (num_steps[i] + 1e-8)
                          for i in range(3)])
    step_size[step_size == 0] = 9999999

    x_steps = np.arange(coord_start[0], coord_end[0] + 1e-8, step_size[0], dtype=np.int)
    y_steps = np.arange(coord_start[1], coord_end[1] + 1e-8, step_size[1], dtype=np.int)
    z_steps = np.arange(coord_start[2], coord_end[2] + 1e-8, step_size[2], dtype=np.int)

    result = torch.zeros([num_classes] + list(padding_shape)).to(device)
    result_n = torch.zeros_like(result).to(device)

    if verbose:
        print('Image Shape: {} Patch Size: {}'.format(padding_shape, patch_size))
        print('X step: %d Y step: %d Z step: %d' %
              (len(x_steps), len(y_steps), len(z_steps)))

    # W H D C =>  C W H D => N C W H D for model input
    input = torch.from_numpy(to_tensor(input)[None]).to(device)

    patchs_slices = []
    for x in x_steps:
        x_mix = x - patch_size[0] // 2
        x_max = x + patch_size[0] // 2
        for y in y_steps:
            y_min = y - patch_size[1] // 2
            y_max = y + patch_size[1] // 2
            for z in z_steps:
                z_min = z - patch_size[2] // 2
                z_max = z + patch_size[2] // 2
                patchs_slices.append([slice(x_mix, x_max),
                                      slice(y_min, y_max),
                                      slice(z_min, z_max)])

    # predict loop
    predict_loop = tqdm(patchs_slices) if verbose else patchs_slices
    model.eval()
    with torch.no_grad():
        for slices in predict_loop:
            output = model(input[[slice(None), slice(None)]+slices])

            if num_classes == 1:
                output = torch.sigmoid(output)
            else:
                output = torch.softmax(output, dim=1)

            result[[slice(None)]+slices] += output[0]
            result_n[[slice(None)]+slices] += 1

    # merge all patchs
    if verbose:
        print('Merging all patchs...')
    result = result / result_n

    if one_hot:
        result = to_numpy(result.cpu().numpy()).astype(np.float32)
    else:
        if num_classes == 1:
            result = torch.squeeze(result, dim=0)
        else:
            result = torch.softmax(result, dim=0)
            result = torch.argmax(result, axis=0)

        result = np.round(result.cpu().numpy()).astype(np.uint8)

    return crop_pad(result, origial_shape)


def predict_case(case,
                 model,
                 target_spacing,
                 normalize_stats,
                 num_classes=3,
                 patch_size=(96, 96, 96),
                 step_per_patch=4,
                 verbose=True,
                 one_hot=False):
    orig_shape = case['image'].shape[:-1]
    affine = case['affine']

    # resample case for predict
    if verbose:
        print('Resampling the case for prediction...')
    case_ = resample_normalize_case(case, target_spacing, normalize_stats)

    if verbose:
        print('Predicting the case...')
    pred = predict_per_patch(case_['image'],
                             model,
                             num_classes,
                             patch_size,
                             step_per_patch,
                             verbose,
                             one_hot)
    if verbose:
        print('Resizing the case to origial shape...')
    case['pred'] = resize(pred, orig_shape, is_label=one_hot is False)
    case['affine'] = affine
    if verbose:
        print('All done!')
    return case


def batch_predict_case(load_dir,
                       save_dir,
                       model,
                       target_spacing,
                       normalize_stats,
                       num_classes=3,
                       patch_size=(240, 240, 80),
                       step_per_patch=4,
                       data_range=None):

    load_dir = Path(load_dir)

    cases = CaseDataset(load_dir, load_meta=True)
    if data_range is None:
        data_range = range(len(cases))

    for i in tqdm(data_range):
        case = predict_case(cases[i],
                            model,
                            target_spacing,
                            normalize_stats,
                            num_classes,
                            patch_size,
                            step_per_patch,
                            False)
        save_pred(case, save_dir)


def cascade_predict_case(case,
                         coarse_model,
                         coarse_target_spacing,
                         coarse_normalize_stats,
                         coarse_patch_size,
                         detail_model,
                         detail_target_spacing,
                         detail_normalize_stats,
                         detail_patch_size,
                         num_classes=3,
                         step_per_patch=4,
                         region_threshold=10000,
                         crop_padding=20,
                         verbose=True):
    if verbose:
        print('Predicting the rough shape for further prediction...')
    case = predict_case(case,
                        coarse_model,
                        coarse_target_spacing,
                        coarse_normalize_stats,
                        1,
                        coarse_patch_size,
                        step_per_patch,
                        verbose=verbose)
    regions = regions_crop_case(case, region_threshold, crop_padding, 'pred')
    num_classes = detail_model.out_channels
    orig_shape = case['image'].shape[:-1]
    result = np.zeros(list(orig_shape)+[num_classes])
    result_n = np.zeros_like(result)
    if verbose:
        print('Cropping regions (%d)...' % len(regions))
    for idx, region in enumerate(regions):
        bbox = region['bbox']
        shape = region['image'].shape[:-1]

        if verbose:
            print('Region {} {} predicting...'.format(idx, shape))
        region = predict_case(region,
                              detail_model,
                              detail_target_spacing,
                              detail_normalize_stats,
                              num_classes,
                              detail_patch_size,
                              step_per_patch,
                              verbose=verbose,
                              one_hot=True)

        region_slices = []
        result_slices = []

        for i in range(len(bbox)):
            region_slice_min = 0 + max(0 - bbox[i][0], 0)
            region_slice_max = shape[i] - max(bbox[i][1] - orig_shape[i], 0)
            region_slices.append(slice(region_slice_min, region_slice_max))

            origin_slice_min = max(bbox[i][0], 0)
            origin_slice_max = min(bbox[i][1], orig_shape[i])
            result_slices.append(slice(origin_slice_min, origin_slice_max))

        region_slices.append(slice(None))
        result_slices.append(slice(None))
        result[result_slices] += region['pred'][region_slices]
        result_n[result_slices] += 1

    if verbose:
        print('Merging all regions...')

    # avoid orig_pred_n = 0
    mask = np.array(result_n > 0)
    result[mask] = result[mask] / result_n[mask]

    if num_classes == 1:
        result = np.squeeze(result, axis=-1)
        result = np.around(result)
    else:
        result = spe.softmax(result, axis=-1)
        result = np.argmax(result, axis=-1)

    case['pred'] = result.astype(np.uint8)
    if verbose:
        print('All done!')
    return case


def cascade_predict(image_file,
                    coarse_model,
                    coarse_target_spacing,
                    coarse_normalize_stats,
                    coarse_patch_size,
                    detail_model,
                    detail_target_spacing,
                    detail_normalize_stats,
                    detail_patch_size,
                    air=-200,
                    num_classes=3,
                    step_per_patch=4,
                    region_threshold=10000,
                    crop_padding=20,
                    label_file=None,
                    verbose=True):

    orig_case = load_case(image_file, label_file)
    case = orient_crop_case(orig_case, air)

    case = cascade_predict_case(case,
                                coarse_model,
                                coarse_target_spacing,
                                coarse_normalize_stats,
                                coarse_patch_size,
                                detail_model,
                                detail_target_spacing,
                                detail_normalize_stats,
                                detail_patch_size,
                                num_classes,
                                step_per_patch,
                                region_threshold,
                                crop_padding,
                                verbose)

    orient = nib.orientations.io_orientation(orig_case['affine'])
    indices = orient[:, 0].astype(np.int)
    orig_shape = np.array(orig_case['image'].shape[:3])
    orig_shape = np.take(orig_shape, indices)
    bbox = case['bbox']

    orig_pred = np.zeros(orig_shape, dtype=np.uint8)
    result_slices = []
    for i in range(len(bbox)):
        orig_slice_min = max(bbox[i][0], 0)
        orig_slice_max = min(bbox[i][1], orig_shape[i])
        result_slices.append(slice(orig_slice_min, orig_slice_max))
    orig_pred[result_slices] = case['pred']

    # orient
    orig_case['pred'] = nib.orientations.apply_orientation(orig_pred, orient)
    if len(orig_case['image'].shape) == 3:
        orig_case['image'] = np.expand_dims(orig_case['image'], -1)

    return orig_case


def batch_cascade_predict(image_dir,
                          save_dir,
                          coarse_model,
                          coarse_target_spacing,
                          coarse_normalize_stats,
                          coarse_patch_size,
                          detail_model,
                          detail_target_spacing,
                          detail_normalize_stats,
                          detail_patch_size,
                          air=-200,
                          num_classes=3,
                          step_per_patch=4,
                          region_threshold=10000,
                          crop_padding=20,
                          data_range=None):

    image_dir = Path(image_dir)
    image_files = [path for path in sorted(image_dir.iterdir()) if path.is_file()]

    if data_range is None:
        data_range = range(len(image_files))

    for i in tqdm(data_range):
        case = cascade_predict(image_files[i],
                               coarse_model,
                               coarse_target_spacing,
                               coarse_normalize_stats,
                               coarse_patch_size,
                               detail_model,
                               detail_target_spacing,
                               detail_normalize_stats,
                               detail_patch_size,
                               air,
                               num_classes,
                               step_per_patch,
                               region_threshold,
                               crop_padding,
                               None,
                               False)
        save_pred(case, save_dir)


def evaluate_case(case):
    num_classes = case['label'].max()
    evaluate_result = []
    for c in range(num_classes):
        pred = np.array(case['pred'] == c+1).astype(np.float32)
        label = np.array(case['label'] == c+1).astype(np.float32)
        dsc = dice(torch.tensor(pred), torch.tensor(label)).item()
        evaluate_result.append(dsc)
    return evaluate_result


def evaluate(label_file, pred_file):

    label_nib = nib.load(str(label_file))
    pred_nib = nib.load(str(pred_file))

    case = {}
    case['label'] = label_nib.get_fdata().astype(np.uint8)
    case['pred'] = pred_nib.get_fdata().astype(np.uint8)
    evaluate_result = evaluate_case(case)
    return evaluate_result


def batch_evaluate(label_dir, pred_dir, data_range=None):
    label_dir = Path(label_dir)
    pred_dir = Path(pred_dir)

    label_files = sorted(list(label_dir.glob('*.nii.gz')))
    pred_files = sorted(list(pred_dir.glob('*.nii.gz')))

    if data_range is None:
        data_range = range(len(label_files))

    evaluate_results = []
    par = tqdm(data_range)
    for i in par:

        evaluate_result = evaluate(label_files[i], pred_files[i])
        evaluate_results.append(evaluate_result)

        evaluate_dict = {}

        for idx, e in enumerate(evaluate_result):
            evaluate_dict["label_%d" % (idx+1)] = e

        par.set_description("Case %d" % i)
        par.set_postfix(evaluate_dict)

    print('\nThe mean dsc of each label:')
    means = np.array(evaluate_results).mean(axis=0)
    for i, mean in enumerate(means):
        print("label_%d: %f" % (i+1, mean))
    return evaluate_results


class Subset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, transform):
        super(Subset, self).__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        case = self.dataset[self.indices[idx]]
        if self.transform:
            case = self.transform(case)
        return case


class Trainer():
    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 dataset,
                 batch_size=10,
                 dataloader_kwargs={'num_workers': 2,
                                    'pin_memory': True},
                 valid_split=0.2,
                 num_samples=None,
                 metrics=None,
                 scheduler=None,
                 train_transform=None,
                 valid_transform=None):

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset
        self.metrics = metrics
        self.scheduler = scheduler
        self.train_transform = train_transform
        self.valid_transform = valid_transform

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(valid_split * dataset_size))
        np.random.shuffle(indices)
        self.train_indices = indices[split:]
        self.valid_indices = indices[:split]

        self.dataloader_kwargs = {'batch_size': batch_size, **dataloader_kwargs}
        self.num_samples = num_samples
        self.valid_split = valid_split
        self.device = next(model.parameters()).device
        self.best_result = {'loss': float('inf')}
        self.current_epoch = 0
        self.patience_counter = 0
        self.amp_state_dict = None

    def get_lr(self, idx=0):
        return self.optimizer.param_groups[idx]['lr']

    def set_lr(self, lr, idx=0):
        self.optimizer.param_groups[idx]['lr'] = lr

    def summary(self, input_shape):
        return summary(self.model, input_shape)

    def batch_loop(self, data_loader, is_train=True):
        results = []
        self.progress_bar.reset(len(data_loader))
        desc = "Epoch %d/%d (LR %.2g)" % (self.current_epoch+1,
                                          self.num_epochs,
                                          self.get_lr())

        self.progress_bar.set_description(desc)
        for batch_idx, batch in enumerate(data_loader):
            x = batch['image'].to(self.device)
            y = batch['label'].to(self.device)

            # forward
            if is_train:
                self.model.train()
                y_pred = self.model(x)
            else:
                self.model.eval()
                with torch.no_grad():
                    y_pred = self.model(x)

            loss = self.loss(y_pred, y)

            # backward
            if is_train:
                self.optimizer.zero_grad()
                if self.use_amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

            result = {'loss': loss.item()}

            # calc the other metrics
            if self.metrics is not None:
                for key, metric_fn in self.metrics.items():
                    result[key] = metric_fn(y_pred, y).item()

            if not torch.isnan(loss):
                results.append(result)

            self.progress_bar.set_postfix(result)
            self.progress_bar.update()

        mean_result = {}
        for key in results[0].keys():
            mean_result[key] = np.mean(np.array([x[key] for x in results]))

        name = 'train' if is_train else 'valid'
        if self.save_dir is not None:
            writer = SummaryWriter(self.save_dir)
            for key in mean_result.keys():
                writer.add_scalar('%s/%s' % (key, name),
                                  mean_result[key],
                                  self.current_epoch)
            writer.close()
        return mean_result

    def fit(self,
            num_epochs=10,
            save_dir=None,
            use_amp=False,
            opt_level='O1'):

        # ----------------------
        #       initialize
        # ----------------------
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.save_dir = save_dir

        if use_amp:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=opt_level)
            if self.amp_state_dict is not None:
                amp.load_state_dict(self.amp_state_dict)

        self.progress_bar = tqdm(total=0)

        # ----------------------
        #      prepare data
        # ----------------------
        train_set = Subset(self.dataset, self.train_indices, self.train_transform)
        if self.num_samples is not None:
            sampler = torch.utils.data.RandomSampler(train_set, True, self.num_samples)
            train_loader = torch.utils.data.DataLoader(train_set,
                                                       sampler=sampler,
                                                       **self.dataloader_kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(train_set,
                                                       shuffle=True,
                                                       **self.dataloader_kwargs)

        if len(self.valid_indices) > 0:
            valid_set = Subset(self.dataset, self.valid_indices, self.valid_transform)
            if self.num_samples is not None:
                num_samples = round(self.num_samples * self.valid_split)
                sampler = torch.utils.data.RandomSampler(valid_set, True, num_samples)
                valid_loader = torch.utils.data.DataLoader(valid_set,
                                                           sampler=sampler,
                                                           **self.dataloader_kwargs)
            else:
                valid_loader = torch.utils.data.DataLoader(valid_set,
                                                           **self.dataloader_kwargs)
        else:
            valid_loader = None

        # ----------------------
        #      main loop
        # ----------------------
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # train loop
            result = self.batch_loop(train_loader, is_train=True)

            # vaild loop
            if valid_loader is not None:
                result = self.batch_loop(valid_loader, is_train=False)

            # build-in fn: lr_scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(result['loss'])
                else:
                    self.scheduler.step()

            # save best
            if result['loss'] < self.best_result['loss']-1e-3:
                self.best_result = result

                if save_dir is not None:
                    self.save_checkpoint(save_dir+'-best.pt')

            if save_dir is not None:
                self.save_checkpoint(save_dir+'-last.pt')

        self.progress_bar.close()

    def save_checkpoint(self, file_path):
        checkpoint = {'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'current_epoch': self.current_epoch,
                      'train_indices': self.train_indices,
                      'valid_indices': self.valid_indices,
                      'best_result': self.best_result}

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp:
            checkpoint['amp_state_dict'] = amp.state_dict()
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch']+1
        self.train_indices = checkpoint['train_indices']
        self.valid_indices = checkpoint['valid_indices']
        self.best_result = checkpoint['best_result']

        if 'amp_state_dict' in checkpoint:
            self.amp_state_dict = checkpoint['amp_state_dict']

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# cross valid
# elif num_folds > 1:
#     # split the dataset into k-fold
#     fold_len = len(dataset) // num_folds
#     fold_len_list = []
#     for i in range(num_folds-1):
#         fold_len_list.append(fold_len)
#     fold_len_list.append(len(dataset)-fold_len * (num_folds-1))
#     fold_subsets = torch.utils.data.random_split(dataset, fold_len_list)

#     fold_metrics = []
#     avg_metrics = {}
#     self.save('init.pt')
#     for i, fold_subset in enumerate(fold_subsets):
#         train_subsets = fold_subsets.copy()
#         train_subsets.remove(fold_subset)
#         train_subset = torch.utils.data.ConcatDataset(train_subsets)

#         train_set = DatasetFromSubset(train_subset, tr_transform)
#         valid_set = DatasetFromSubset(fold_subset, vd_transform)

#         print('Fold %d/%d:' % (i+1, num_folds))
#         self.load('init.pt')
#         train_kwargs['log_dir'] = '%s_%d' % (log_dir, i)
#         metrics = self.train(train_set, valid_set, **train_kwargs)
#         fold_metrics.append(metrics)

#     # calc the avg
#     for name in fold_metrics[0].keys():
#         sum_metric = 0
#         for fold_metric in fold_metrics:
#             sum_metric += fold_metric[name]

#         avg_metrics[name] = sum_metric / num_folds

#     for i, fold_metric in enumerate(fold_metrics):
#         print('Fold %d metrics:\t%s' %
#                 (i+1, self.metrics_stringify(fold_metric)))
#     print('Avg metrics:\t%s' % self.metrics_stringify(avg_metrics))

# manual ctrl @lr_factor @min_lr @patience
# if metrics['Loss'] < best_metrics['Loss']-1e-4:
#     if save_dir and save_best:
#         self.save(save_dir+'-best.pt')
#     best_metrics = metrics
#     patience_counter = 0

# elif patience > 0:
#     patience_counter += 1
#     if patience_counter > patience:
#         print("│\n├Loss stopped improving for %d num_epochs." %
#               patience_counter)

#         patience_counter = 0
#         lr = self.get_lr() * lr_factor
#         if min_lr and lr < min_lr:
#             print("│LR below the min LR, stop training.")
#             break
#         else:
#             print('│Reduce LR to %.3g' % lr)
#             self.set_lr(lr)

# def get_lr(self):
#         for param_group in self.optimizer.param_groups:
#             return param_group['lr']

# def set_lr(self, lr):
#     for param_group in self.optimizer.param_groups:
#         param_group['lr'] = lr

# # save best & early_stop_patience counter
# if result['loss'] < self.best_result['loss']-1e-3:
#     self.best_result = result
#     self.patience_counter = 0

#     if save_dir and save_best:
#         self.save_checkpoint(save_dir+'-best.pt')

# elif early_stop_patience > 0:
#     self.patience_counter += 1
#     if self.patience_counter > early_stop_patience:
#         print(("\nLoss stopped improving for %d num_epochs. "
#                 "stop training.") % self.patience_counter)
#         self.patience_counter = 0
#         break
