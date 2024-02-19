import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from loss import NetPose
from trainer import Trainer
from models import Fusion
from datasets import MultiPose

# 1e-3, 扩大范围的wingloss，batch size 128
# 2e-4, 进一步降低loss，batch size 256

cfg = {
    'root_path': '/data/human36m/processed',
    'labels_path': '/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy',
    'lr':2e-4,
    'num_epoch':300,
    'batch_size_train': 256,
    'batch_size_test': 64,
    'num_workers':8,
    'num_keypoints': 17,
    'num_views':4,
    'image_size':384,
    'heatmap_size':96,
    'device':'cuda',
    'model_path': '/logs/tri/model_125.pth',
    'save_dir': '/logs/tri',
}

train_db = MultiPose(cfg, is_train=True)
test_db = MultiPose(cfg, is_train=False)
train_loader = DataLoader(
    train_db,
    batch_size=cfg['batch_size_train'], 
    shuffle=True,
    num_workers = cfg['num_workers'],
    pin_memory = True,
    drop_last=True,
)
test_loader = DataLoader(
    test_db,
    batch_size=cfg['batch_size_test'], 
    shuffle=False,
    num_workers = cfg['num_workers'],
    pin_memory = True,
    drop_last=True,
)



model = Fusion(cfg)

if 'model_path' in cfg:
    pretrain_dict = torch.load(cfg['model_path'])
    missing, unexpected = model.load_state_dict(pretrain_dict,strict=False)
    print('missing length', len(missing), 'unexpected', len(unexpected) , '\n')
# model = torch.nn.DataParallel(model, device_ids=[0,1,2]).cuda()
model = model.cuda()

net = NetPose(cfg, model)
trainer = Trainer(cfg, net)
trainer.run(train_loader, test_loader)