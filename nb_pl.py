# %%
from tqdm import tqdm, trange
from visualize import grid_plt, sample_plt
from data import CaseDataset, load_crop, resample_normalize
from trainer import DatasetFromSubset
from network import generate_paired_features, Unet, ResBlock, ResBlockStack
from loss import DiceCoef, FocalDiceCoefLoss, dice_coef

from torchvision.transforms import Compose
from transform import Crop, resize, rescale, to_one_hot, RandomCrop, ToOnehot, ToNumpy, ToTensor, \
    CombineLabels, RandomBrightness, RandomContrast, RandomGamma, RandomRescale, RandomRescaleCrop, \
    RandomMirror, pad, crop_pad, to_tensor, to_numpy


import torch
import torch.nn as nn
import torch.optim as optim
from apex import amp

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, LightningModule

from datetime import datetime
import numpy as np
import nibabel as nib
from utils import json_load, json_save
from pathlib import Path
from argparse import Namespace, ArgumentParser


args = Namespace(data_set_dir='data/Task00_Kidney/normalized',
                 learning_rate=1e-4,
                 num_workers=2,
                 batch_size=1,
                 valid_split=0.2,
                 patch_x=160,
                 patch_y=160,
                 patch_z=80,
                 num_features=30,
                 num_pool=4)


class Unet3D(LightningModule):

    def __init__(self, hparams):
        super(Unet3D, self).__init__()
        self.hparams = hparams
        self.learning_rate = hparams.learning_rate
        self.data_set_dir = hparams.data_set_dir
        self.loader_kwargs = {'batch_size': hparams.batch_size,
                              'num_workers': hparams.num_workers,
                              'pin_memory': True}
        self.valid_split = hparams.valid_split

        num_pool = hparams.num_pool
        num_features = hparams.num_features
        patch_size = (hparams.patch_x,
                      hparams.patch_y,
                      hparams.patch_z)

        def encode_kwargs_fn(level):
            num_stacks = max(level, 1)
            return {'num_stacks': num_stacks}

        paired_features = generate_paired_features(num_pool, num_features)

        self.net = Unet(in_channels=1,
                        out_channels=1,
                        paired_features=paired_features,
                        pool_block=ResBlock,
                        pool_kwargs={'stride': 2},
                        up_kwargs={'attention': True},
                        encode_block=ResBlockStack,
                        encode_kwargs_fn=encode_kwargs_fn,
                        decode_block=ResBlock)

        self.loss_fn = FocalDiceCoefLoss()
        self.metrics = {'kd_dsc': DiceCoef()}

        self.tr_transform = Compose([
            RandomRescaleCrop(0.1,
                              patch_size,
                              crop_mode='random',
                              enforce_label_indices=[1]),
            RandomMirror((0.2, 0, 0)),
            RandomContrast(0.1),
            RandomBrightness(0.1),
            RandomGamma(0.1),
            CombineLabels([1, 2], 3),
            ToTensor()
        ])
        self.vd_transform = Compose([
            RandomCrop(patch_size),
            CombineLabels([1, 2], 3),
            ToTensor()
        ])

    def prepare_data(self):
        data_set = CaseDataset(self.data_set_dir)
        n_valid = round(len(data_set) * self.valid_split)
        valid_subset, train_subset = torch.utils.data.random_split(
            data_set, [n_valid, len(data_set)-n_valid])
        self.train_set = DatasetFromSubset(train_subset, self.tr_transform)
        self.valid_set = DatasetFromSubset(valid_subset, self.vd_transform)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def run_step(self, batch, prefix=''):
        res_dict = {}
        input, target = batch['image'], batch['label']
        output = self.forward(input)
        res_dict[prefix+'loss'] = self.loss_fn(output, target)

        for name, metric_fn in self.metrics.items():
            res_dict[prefix+name] = metric_fn(output, target)

        return res_dict

    def on_save_checkpoint(self, checkpoint):
        checkpoint['amp'] = amp.state_dict()

    def on_load_checkpoint(self, checkpoint):
        print(checkpoint['amp'])
        amp.load_state_dict(checkpoint['amp'])

    def training_step(self, batch, batch_idx):
        res_dict = self.run_step(batch)
        tqdm_dict = res_dict.copy()
        del tqdm_dict['loss']
        return {'loss': res_dict['loss'],
                'progress_bar': tqdm_dict,
                'log': res_dict}

    def validation_step(self, batch, batch_idx):
        return self.run_step(batch, prefix='val_')

    def validation_epoch_end(self, outputs):
        val_mean = {}
        for name in outputs[0].keys():
            val_mean[name] = torch.stack([x[name] for x in outputs]).mean()
        return {'progress_bar': val_mean,
                'log': val_mean}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set,
                                           shuffle=True,
                                           **self.loader_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set,
                                           **self.loader_kwargs)

    def predict_3d(self, input, patch_size, out_channels, step=2):
        is_cuda = next(self.parameters()).is_cuda
        device = torch.device('cuda:0' if is_cuda else 'cpu')

        print('Tile generating pred...')
        # W H D
        orig_shape = input.shape[:3]
        print('Data original shape: (%d, %d, %d)' % (orig_shape[0], orig_shape[1], orig_shape[2]))
        input = pad(input, patch_size)
        pad_shape = input.shape[:3]
        print('Data padding shape: (%d, %d, %d)' %
              (pad_shape[0], pad_shape[1], pad_shape[2]))

        coord_start = np.array([i // 2 for i in patch_size]).astype(int)
        coord_end = np.array(
            [pad_shape[i] - patch_size[i] // 2 for i in range(len(patch_size))]).astype(int)
        num_steps = np.ceil(
            [(coord_end[i] - coord_start[i]) / (patch_size[i] / step) for i in range(3)])
        step_size = np.array(
            [(coord_end[i] - coord_start[i]) / (num_steps[i] + 1e-8) for i in range(3)])
        step_size[step_size == 0] = 9999999

        xsteps = np.round(np.arange(
            coord_start[0], coord_end[0] + 1e-8, step_size[0])).astype(int)
        ysteps = np.round(np.arange(
            coord_start[1], coord_end[1] + 1e-8, step_size[1])).astype(int)
        zsteps = np.round(np.arange(
            coord_start[2], coord_end[2] + 1e-8, step_size[2])).astype(int)

        result = torch.zeros([out_channels] + list(pad_shape)).to(device)
        result_n = torch.zeros([out_channels] + list(pad_shape)).to(device)
        n_add = torch.ones([out_channels] + list(patch_size)).to(device)
        print('┌X step: %d\n├Y step: %d\n└Z step: %d' % (len(xsteps), len(ysteps), len(zsteps)))

        # W H D C =>  C W H D => N C W H D for model input
        input = to_tensor(input)[None]
        input = torch.from_numpy(input).to(device)
        self.eval()
        with torch.no_grad():
            for x in xsteps:
                x_s = x - patch_size[0] // 2
                x_e = x + patch_size[0] // 2
                for y in ysteps:
                    y_s = y - patch_size[1] // 2
                    y_e = y + patch_size[1] // 2
                    for z in zsteps:
                        z_s = z - patch_size[2] // 2
                        z_e = z + patch_size[2] // 2
                        print('Predict on patch-%d-%d-%d' % (x, y, z))
                        output = self.forward(input[:, :, x_s:x_e, y_s:y_e, z_s:z_e])
                        if out_channels == 1:
                            output = torch.sigmoid(output)
                        else:
                            output = torch.softmax(output, dim=1)
                        # N C W H D => C W H D
                        result[:, x_s:x_e, y_s:y_e, z_s:z_e] += output[0]
                        result_n[:, x_s:x_e, y_s:y_e, z_s:z_e] += n_add

        print('Merge all patchs')
        result = result / result_n
        if out_channels == 1:
            result.squeeze_(dim=0)
            result = np.round(result.cpu().numpy()).astype(np.uint8)
        else:
            result = torch.softmax(result, dim=0)
            # C W H D => W H D  one-hot to label
            result = torch.argmax(result, axis=0).cpu().numpy().astype(np.uint8)
        return crop_pad(result, orig_shape)

    def prepare(self, image_file, props_file, save_dir=None):

        props_file = Path(props_file)
        props = json_load(str(props_file))

        air = props['air']
        spacing = props['resampled_spacing']
        statstics = props['normalize_statstics']

        print('Cropping image...')
        image, _, meta = load_crop(image_file, None, air=air)
        print('Resample image to target spacing...')
        image, _, meta = resample_normalize(image,
                                            None,
                                            meta,
                                            spacing=spacing,
                                            statstics=statstics)

        if save_dir:
            save_dir = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            data_fname = '%s_data.npz' % meta['case_id']
            np.savez(str(save_dir / data_fname), image=image)

            meta_fname = '%s_meta.json' % meta['case_id']
            json_save(str(save_dir / meta_fname), meta)

        return {'image': image, 'meta': meta}

    def rebuild_pred(self, pred, meta, save_dir=None):
        affine = meta['affine']
        cropped_shape = meta['cropped_shape']
        original_shape = meta['shape']
        orient = meta['orient']

        # pad_width for np.pad
        pad_width = meta['nonair_bbox']
        for i in range(len(original_shape)):
            pad_width[i][1] = original_shape[i] - (pad_width[i][1] + 1)

        print('Resample pred to original spacing...')
        pred = resize(pred, cropped_shape, is_label=True)
        print('Add padding to pred...')
        pred = np.pad(pred, pad_width, constant_values=0)
        pred = nib.orientations.apply_orientation(pred, orient)

        if save_dir:
            save_dir = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            pred_nib = nib.Nifti1Pair(pred, np.array(affine))
            nib_fname = '%s_pred.nii.gz' % meta['case_id']
            nib.save(pred_nib, str(save_dir / nib_fname))

        return {'pred': pred, 'meta': meta}

    def generate_pred(self, out_channels, image_file, props_file, patch_size, save_dir=None):
        sample = self.prepare(image_file=image_file,
                              props_file=props_file,
                              save_dir=save_dir)
        pred = self.predict_3d(sample['image'], patch_size, out_channels)
        sample_pred = self.rebuild_pred(pred, sample['meta'], save_dir=save_dir)
        print('All Done!')
        return sample_pred

    def generate_paired_data(self,
                             out_channels,
                             image_file,
                             label_file,
                             save_dir,
                             props_file,
                             patch_size):
        sample = self.generate_pred(out_channels=out_channels,
                                    image_file=image_file,
                                    props_file=props_file,
                                    patch_size=patch_size,
                                    save_dir=save_dir)
        image = nib.load(image_file).get_fdata().astype(np.float32)
        sample['image'] = np.expand_dims(image, axis=-1)
        if label_file:
            sample['label'] = nib.load(label_file).get_fdata().astype(np.uint8)
        return sample


model = Unet3D(hparams=args)

# %%
# version = datetime.now().strftime("%y%m%d%H%H%M%S")
# logger = TensorBoardLogger('logs', name='Task00_Kidney_00', version=version)
# checkpoint = ModelCheckpoint('logs/Task00_Kidney_00/%s' % version)
# early_stop = EarlyStopping(patience=100, min_delta=1e-3)
# 'logs/Task00_Kidney_00/lightning_logs/version_0/checkpoints/epoch=7.ckpt'
# 'logs/Task00_Kidney_00/'
resume_ckpt = 'logs/Task00_Kidney_00/lightning_logs/version_14/checkpoints/epoch=81.ckpt'
save_path = 'logs/Task00_Kidney_00/'

trainer = Trainer(gpus=1,
                  amp_level='O2',
                  precision=16,
                  progress_bar_refresh_rate=1,
                  train_percent_check=1,
                  max_epochs=500,
                  min_epochs=100,
                  #   logger=logger,
                  #   checkpoint_callback=checkpoint,
                  #   early_stop_callback=early_stop,
                  default_save_path=save_path,
                  resume_from_checkpoint=resume_ckpt)
trainer.fit(model)


# %%
resume_ckpt = 'logs/Task00_Kidney_00/lightning_logs/version_10/checkpoints/epoch=73.ckpt'
model = Unet3D.load_from_checkpoint(resume_ckpt).cuda()
# %%
sample = CaseDataset('data/Task00_Kidney/normalized')[7]
sample['pred'] = model.predict_3d(sample['image'], (160, 160, 80), 1)
# %%
image_file = '/mnt/main/dataset/Task00_Kidney/imagesTr/case_00008.nii.gz'
label_file = '/mnt/main/dataset/Task00_Kidney/labelsTr/case_00008.nii.gz'
save_dir = 'data/Task00_Kidney/test/'
props_file = 'data/Task00_Kidney/props.json'

sample = model.generate_paired_data(out_channels=1,
                                    image_file=image_file,
                                    label_file=label_file,
                                    props_file=props_file,
                                    patch_size=(160, 160, 80),
                                    save_dir=save_dir)
# %%
sample_plt(sample, slice_pct=0.7, axi=0)


# %%
resume_ckpt = 'logs/Task00_Kidney_00/lightning_logs/version_16/checkpoints/epoch=0.ckpt'
resume_ckpt2 = 'logs/Task00_Kidney_00/lightning_logs/version_14/checkpoints/epoch=81.ckpt'
checkpoint = torch.load(resume_ckpt)
checkpoint2 = torch.load(resume_ckpt2)
# amp_dict = checkpoint['amp']
# checkpoint2['amp'] = amp_dict.copy()
# checkpoint2['amp']['loss_scaler0']['loss_scale'] = 1.0

print(checkpoint2['amp'])
# torch.save(checkpoint2, resume_ckpt2)

# %%
