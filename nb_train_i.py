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

model = ResUnet3D().cuda()
optimizer = Adam(model.parameters(), lr=1e-5)
loss = FocalDiceCoefLoss()
metrics = {'kd_dsc': Dice()}
scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=60)
dataset = CaseDataset('data/Task00_Kidney/norm')
patch_size = (144, 144, 96)
train_transform = Compose([
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

valid_transform = Compose([
    RandomCrop(patch_size),
    CombineLabels([1, 2], 3),
    ToTensor()
])

ckpt = torch.load('logs/Task00_Kidney/kd-2004052152-epoch=314.pt')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])

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
save_dir = "logs/Task00_Kidney/kd-{}".format(datetime.now().strftime("%y%m%d%H%M"))
# save_dir = 'logs/Task00_Kidney/kd-2003240211'
# trainer.load_checkpoint('logs/Task00_Kidney/kd-2003240211-epoch=260.ckpt')
trainer.fit(
    num_epochs=800,
    use_amp=True,
    save_dir=save_dir
)


# %%
