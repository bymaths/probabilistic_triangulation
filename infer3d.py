import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from utils import *
from trainer import AverageMeter
from datasets import MultiHuman36M
from models import ProbTri
from loss import Net3d
from tqdm import tqdm

cfg = {
    'root_path': '/data/human36m/processed',
    'labels_path': '/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy',
    'lr':1e-5,
    'num_epoch':300,
    'batch_size_train': 32,
    'batch_size_test': 8,
    'num_workers':8,
    'num_keypoints': 17,
    'num_views': 4,
    'scaleRange': [1.1,1.2],
    'moveRange': [-0.1,0.1],
    'image_size':384,
    'heatmap_size': 96,
    # 'backbone_path': 'checkpoints/backbone.pth',
    # 'fusion_path': 'checkpoints/fusion.pth',
    'model_path': 'checkpoints/pretrain.pth',
    'device':'cuda',
    'save_dir': '/logs/pose3d',
    'use_tag':False,
    'data_skip_train':8,
    'data_skip_test':1,
}

test_db = MultiHuman36M(cfg, is_train=False)
test_loader = DataLoader(
    test_db,
    batch_size=cfg['batch_size_test'], 
    shuffle=False,
    num_workers = cfg['num_workers'],
    pin_memory = True,
    drop_last=True,
)

# trainer
model = ProbTri(cfg).cuda()
if 'backbone_path' in cfg:
    pretrain_dict = torch.load(cfg['backbone_path'])
    missing, unexpected = model.backbone.load_state_dict(pretrain_dict,strict=False)
    print('load backbone model, missing length', len(missing), 'unexpected', len(unexpected) , '\n')

if 'fusion_path' in cfg:
    pretrain_dict = torch.load(cfg['fusion_path'])
    missing, unexpected = model.fusion.load_state_dict(pretrain_dict,strict=False)
    print('load fusion model, missing length', len(missing), 'unexpected', len(unexpected) , '\n')

if 'model_path' in cfg:
    pretrain_dict = torch.load(cfg['model_path'])
    missing, unexpected = model.load_state_dict(pretrain_dict,strict=False)
    print('missing length', len(missing), 'unexpected', len(unexpected) , '\n')

net = Net3d(cfg, model)
net.eval()
torch.set_grad_enabled(False)


avg_loss_stats = {}
for iter_id, batch in tqdm(enumerate(test_loader)):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device = cfg['device'], non_blocking=True)

    output, loss, loss_stats = net(batch)
    if 'loss' not in avg_loss_stats:
        for key in loss_stats:
            avg_loss_stats[key] = AverageMeter()
    
    for key in loss_stats:
        avg_loss_stats[key].update(loss_stats[key].item(), test_loader.batch_size)

for k, v in avg_loss_stats.items():
    print(k, v.avg)