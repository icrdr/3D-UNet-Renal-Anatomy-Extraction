# %%
from tqdm import tqdm, trange
from visualize import grid_plt, sample_plt
from data import CaseDataset
from trainer import DatasetFromSubset, Trainer, predict_3d_tile
from network import generate_paired_features, Unet, ResBlock, ResBlockStack
from loss import DiceCoef, FocalDiceCoefLoss, dice_coef

from torchvision.transforms import Compose
from transform import Crop, resize, rescale, to_one_hot, RandomCrop, ToOnehot, ToNumpy, ToTensor, \
    CombineLabels, RandomBrightness, RandomContrast, RandomGamma, RandomRescale, RandomRescaleCrop, \
    RandomMirror, pad, crop_pad, to_tensor, to_numpy

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, LightningModule

from datetime import datetime
from argparse import Namespace, ArgumentParser

parser = ArgumentParser()
parser.add_argument('-data', type=str,
                    default='data/Task00_Kidney/normalized',
                    dest='data_set_dir',
                    help='display an integer')
parser.add_argument('-lr', type=float,
                    default=1e-4,
                    dest='learning_rate',
                    help='display an integer')
parser.add_argument('-pool', type=int,
                    default=4,
                    dest='num_pool',
                    help='display an integer')
parser.add_argument('-feature', type=int,
                    default=30,
                    dest='num_features',
                    help='display an integer')
parser.add_argument('-patch-x', type=int,
                    default=160,
                    dest='patch_x',
                    help='Add repeated values to a list')
parser.add_argument('-patch-y', type=int,
                    default=160,
                    dest='patch_y',
                    help='Add repeated values to a list')
parser.add_argument('-patch-z', type=int,
                    default=80,
                    dest='patch_z',
                    help='Add repeated values to a list')
parser.add_argument('-split', type=float,
                    default=0.2,
                    dest='valid_split',
                    help='Add repeated values to a list')
parser.add_argument('-batch', type=int,
                    default=1,
                    dest='batch_size',
                    help='Add repeated values to a list')
parser.add_argument('-worker', type=int,
                    default=2,
                    dest='num_workers',
                    help='Add repeated values to a list')
parser.add_argument('-resume', type=str,
                    default='',
                    dest='resume_ckpt',
                    help='display an integer')
parser.add_argument('-save', type=str,
                    default='',
                    dest='save_path',
                    help='display an integer')
args = parser.parse_args()
print(args)


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

        self.loss = FocalDiceCoefLoss()

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

    def training_step(self, batch, batch_idx):
        input, target = batch['image'], batch['label']
        output = self.forward(input)
        loss = self.loss(output, target)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        input, target = batch['image'], batch['label']
        output = self.forward(input)
        loss = self.loss(output, target)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set,
                                           shuffle=True,
                                           **self.loader_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set,
                                           **self.loader_kwargs)


model = Unet3D(hparams=args)

# %%
# version = datetime.now().strftime("%y%m%d%H%H%M%S")
# logger = TensorBoardLogger('logs', name='Task00_Kidney_00', version=version)
# checkpoint = ModelCheckpoint('logs/Task00_Kidney_00/%s' % version)
# early_stop = EarlyStopping(patience=100, min_delta=1e-3)
# 'logs/Task00_Kidney_00/lightning_logs/version_0/checkpoints/epoch=7.ckpt'
# 'logs/Task00_Kidney_00/'
resume_ckpt = args.resume_ckpt if args.resume_ckpt else None
save_path = args.save_path if args.ckpt_path else None

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
                  resume_from_checkpoint=resume_ckpt
                  )
trainer.fit(model)
