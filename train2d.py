import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from utils import *
from trainer import Trainer
from datasets import Human36M
from models import pose2d_model
from loss import Net2d
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

cfg = {
    'root_path': '/data/human36m/processed',
    'labels_path': '/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy',
    'lr':1e-4,
    'num_epoch':300,
    'batch_size_train': 128,
    'batch_size_test': 32,
    'num_workers':32,
    'num_keypoints': 17,
    'scaleRange': [1.1,1.2],
    'moveRange': [-0.1,0.1],
    'image_size':384,
    'heatmap_size': 96,
    # 'model_path': 'checkpoints/mobileone_s4_unfused.pth.tar',
    'model_path': '/logs/pose2d_384/model_12.pth',
    # 'model_path': 'checkpoints/backbone_pretrain.pth',
    'device':'cuda',
    'save_dir': '/logs/pose2d_384',
    'use_tag':False,
    'data_skip_train':16,
    'data_skip_test':16,
}

train_db = Human36M(cfg, is_train=True)
test_db = Human36M(cfg, is_train=False)
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


# trainer
model = pose2d_model(num_classes=17)

if 'model_path' in cfg:
    pretrain_dict = torch.load(cfg['model_path'])
    missing, unexpected = model.load_state_dict(pretrain_dict,strict=False)
    print('missing length', len(missing), 'unexpected', len(unexpected) , '\n')
model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()

net = Net2d(cfg, model)
trainer = Trainer(cfg, net)
trainer.run(train_loader, test_loader)
