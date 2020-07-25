# %%
from losses import DiceCoef, FocalDiceCoefLoss
from data_proccess import CaseDataset
from train import Trainer
from network import generate_paired_features, Unet, ResBlock, ResBlockStack

from batchgenerators.transforms.spatial_transforms \
    import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms \
    import BrightnessMultiplicativeTransform, \
    GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.crop_and_pad_transforms \
    import RandomCropTransform
from batchgenerators.transforms import Compose
import torch.optim as optim
import numpy as np
from datetime import datetime

case_set = CaseDataset('./data/Task00_Kidney/preproccess_data')
print(len(case_set))


num_pool = 4
num_features = 30


def encode_kwargs_fn(level):
    num_stacks = max(level, 1)
    return {'num_stacks': num_stacks}


paired_features = generate_paired_features(num_pool, num_features)

model = Unet(in_channels=1,
             out_channels=3,
             paired_features=paired_features,
             pool_block=ResBlock,
             pool_kwargs={'stride': 2},
             up_kwargs={'attention': True},
             encode_block=ResBlockStack,
             encode_kwargs_fn=encode_kwargs_fn,
             decode_block=ResBlock).cuda()


patch_size = (160, 160, 80)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.2, patience=30)

tr_transform = Compose([
    GammaTransform((0.9, 1.1)),
    ContrastAugmentationTransform((0.9, 1.1)),
    BrightnessMultiplicativeTransform((0.9, 1.1)),
    MirrorTransform(axes=[0]),
    SpatialTransform_2(
        patch_size, (90, 90, 50), random_crop=True,
        do_elastic_deform=True, deformation_scale=(0, 0.05),
        do_rotation=True,
        angle_x=(-0.1 * np.pi, 0.1 * np.pi),
        angle_y=(0, 0), angle_z=(0, 0),
        do_scale=True, scale=(0.9, 1.1),
        border_mode_data='constant',
    ),
    RandomCropTransform(crop_size=patch_size),
])

vd_transform = Compose([
    RandomCropTransform(crop_size=patch_size),
])

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=FocalDiceCoefLoss(d_weight=[1, 10, 20]),
    metrics={'Kd Dsc': DiceCoef(weight=[0, 1, 0]), 'Ca Dsc': DiceCoef(weight=[0, 0, 1])},
    scheduler=scheduler,
    tr_transform=tr_transform,
    vd_transform=vd_transform,
)
trainer.save('init.pt')

log_dir = "logs/Task00_Kidney/att-res-{}".format(datetime.now().strftime("%H%M%S"))
trainer.load('logs/Task00_Kidney/att-res-065552-last.pt')
trainer.fit(
    case_set,
    batch_size=1,
    epochs=500,
    # valid_split=0.2,
    num_samples=250,
    log_dir=log_dir,
    save_dir=log_dir,
    save_last=True,
    save_best=True,
    num_workers=2,
    pin_memory=True
)
