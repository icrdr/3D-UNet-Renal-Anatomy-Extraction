# %%
from trainer import Trainer
from network import ResUnet3D
from loss import DiceCoef, FocalDiceCoefLoss
from data import CaseDataset

from torchvision.transforms import Compose
from transform import Crop, RandomCrop, ToTensor, CombineLabels, \
    RandomBrightness, RandomContrast, RandomGamma, \
    RandomRescale, RandomRescaleCrop, RandomMirror

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import torch


model = ResUnet3D(out_channels=3).cuda()
optimizer = Adam(model.parameters(), lr=1e-4)
loss = FocalDiceCoefLoss(d_weight=[1, 10, 20])
metrics = {'kd_dsc': DiceCoef(weight=[0, 1, 0]),
           'ca_dsc': DiceCoef(weight=[0, 0, 1])}
scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=30)
dataset = CaseDataset('data/Task00_Kidney/region_norm')
patch_size = (128, 128, 128)
train_transform = Compose([
    RandomRescaleCrop(0.1,
                      patch_size,
                      crop_mode='random',
                      enforce_label_indices=[1]),
    RandomMirror((0.5, 0.5, 0.5)),
    RandomContrast(0.1),
    RandomBrightness(0.1),
    RandomGamma(0.1),
    ToTensor()
])

valid_transform = Compose([
    RandomCrop(patch_size),
    ToTensor()
])

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
    dataset=dataset,
    scheduler=scheduler,
    train_transform=train_transform,
    valid_transform=valid_transform,
    batch_size=2,
    valid_split=0
)

# %%
save_dir = "logs/Task00_Kidney/ca-{}".format(datetime.now().strftime("%y%m%d%H%M"))
save_dir = 'logs/Task00_Kidney/ca-2004080007'
trainer.load_checkpoint('logs/Task00_Kidney/ca-2004080007-last.pt')

trainer.fit(
    num_epochs=800,
    use_amp=True,
    save_dir=save_dir
)
