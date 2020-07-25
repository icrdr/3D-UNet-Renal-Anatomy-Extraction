# %%
from trainer import Trainer
from network import ResUnet3D, ResAttrUnet3D, ResAttrUnet3D2, ResAttrBNUnet3D
from loss import Dice, HybirdLoss, DiceLoss, FocalLoss
from data import CaseDataset

from torchvision.transforms import Compose
from transform import Crop, RandomCrop, ToTensor, CombineLabels, \
    RandomBrightness, RandomContrast, RandomGamma, \
    RandomRescale, RandomRescaleCrop, RandomMirror

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime


model = ResUnet3D(out_channels=3).cuda()
optimizer = Adam(model.parameters(), lr=1e-4)
loss = HybirdLoss(weight_v=[1, 148, 191], alpha=0.9, beta=0.1)
metrics = {'dsc': DiceLoss(weight_v=[1, 148, 191], alpha=0.9, beta=0.1),
           'focal': FocalLoss(weight_v=[1, 148, 191]),
           'a_dsc': Dice(weight_v=[0, 1, 0]),
           'v_dsc': Dice(weight_v=[0, 0, 1])}
scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=25)
dataset = CaseDataset('data/Task20_Kidney/vessel_region_norm')
patch_size = (128, 128, 128)
train_transform = Compose([
    RandomRescaleCrop(0.1,
                      patch_size,
                      crop_mode='random'),
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

# ckpt = torch.load('logs/Task20_Kidney/av-loss-last.pt')
# model.load_state_dict(ckpt['model_state_dict'])
# optimizer.load_state_dict(ckpt['optimizer_state_dict'])

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
    valid_split=0.0,
    num_samples=200,
)

# %%
save_dir = "logs/DOC/iib-H-09-{}".format(datetime.now().strftime("%y%m%d%H%M"))
save_dir = 'logs/DOC/iib-H-09-2006150257'
trainer.load_checkpoint('logs/DOC/iib-H-09-2006150257-last.pt')
trainer.fit(
    num_epochs=800,
    use_amp=True,
    save_dir=save_dir
)


# %%
