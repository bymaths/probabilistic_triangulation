import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from utils import Camera, eulid_to_homo
import numpy as np
from augment import *
import os
from collections import defaultdict
import random

class Human36M(Dataset):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.labels = np.load(cfg['labels_path'], allow_pickle=True).item()
        # n_cameras = len(self.labels['camera_names'])

        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_subjects = ['S9','S11']
        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)
        test_subjects  = list(self.labels['subject_names'].index(x) for x in test_subjects)

        if is_train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
        else:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)


        self.labels['table'] = self.labels['table'][mask]

        self.augment = Compose([
            Crop(cfg['scaleRange'],cfg['moveRange']),
            Resize(cfg['image_size']),
            PhotometricDistort(),
            NormSkeleton(),
            NormImage(),
            GenHeatmap(cfg['num_keypoints'],cfg['image_size'],cfg['heatmap_size']),
        ])

    def __len__(self):
        if self.is_train:
            return (len(self.labels['table'])*len(self.labels['camera_names']))//self.cfg['data_skip_train'] 
        else:
            return len(self.labels['table'])*len(self.labels['camera_names'])//self.cfg['data_skip_test'] 

    def __getitem__(self, index):
        if self.is_train:
            index = index * self.cfg['data_skip_train']  + np.random.randint(self.cfg['data_skip_train'])
        else:
            index = index * self.cfg['data_skip_test']  + np.random.randint(self.cfg['data_skip_test'])
            
        camera_idx = index % len(self.labels['camera_names'])
        idx = index // len(self.labels['camera_names'])
        shot = self.labels['table'][idx]
        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']
        camera_name = self.labels['camera_names'][camera_idx]

        

        image = cv2.imread(os.path.join(
            self.cfg['root_path'], subject, action, 'imageSequence' + '-undistorted',
            camera_name, 'img_%06d.jpg' % (frame_idx+1)))

        box = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB

        # x3d = np.pad(
        #     shot['keypoints'][:self.cfg['num_keypts']],
        #     ((0,0), (0,1)), 'constant', constant_values=1.0) 
        x3d = np.asarray(shot['keypoints'][:self.cfg['num_keypoints']])

        shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
        camera = Camera(shot_camera['R'],shot_camera['t'],shot_camera['K'])

        sample = self.augment({'image':image, 'box': box, 'x3d': x3d, 'camera': camera})

        if self.cfg['use_tag']:
            sample['tag'] = {
                'subject': shot['subject_idx'],
                'action': shot['action_idx'],
                'camera': camera_idx,
                'frame': frame_idx,
            }

        return sample

class MultiPose(Dataset):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.labels = np.load(cfg['labels_path'], allow_pickle=True).item()
        # n_cameras = len(self.labels['camera_names'])

        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_subjects = ['S9', 'S11']
        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)
        test_subjects  = list(self.labels['subject_names'].index(x) for x in test_subjects)

        if is_train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
        else:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)

        self.labels['table'] = self.labels['table'][mask]

        self.augment = Compose([
            NormSkeleton(),
        ])

    def __len__(self):
        return len(self.labels['table'])

    def __getitem__(self, index):
        shot = self.labels['table'][index]
        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']

        view_list = []
        for camera_idx, camera_name in enumerate(self.labels['camera_names']):
            x3d = np.asarray(shot['keypoints'][:self.cfg['num_keypoints']])
            shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
            camera = Camera(shot_camera['R'],shot_camera['t'],shot_camera['K'])
            view_list.append(self.augment({'x3d': x3d, 'camera': camera}))

        random.shuffle(view_list)
        sample = defaultdict(list)
        for view in view_list:
            for key in view:
                sample[key].append(view[key])
        
        for key in sample:
            sample[key] = np.asarray(sample[key])

        V,J,_ = sample['x2d'].shape
        # (V,1,3,3) @ (V,J,3,1) -> (V,J,3) -> (V,J,4)
        if self.is_train:
            rands = np.random.randn(V,J,2)*2
            x2d = sample['x2d'] + rands
            conf = np.exp(-np.sqrt((rands**2).sum(-1)))[...,None]
        else:
            x2d = sample['x2d'] 
            conf = np.ones((V,J,1))
        # (V,1,3,3) @ (V,J,3,1) -> (V,J,3,1)
        xdir = (np.linalg.inv(sample['K'])[:,None] @ eulid_to_homo(x2d)[...,None]).squeeze(-1)
        xdir = np.concatenate([xdir, conf], axis=-1).astype(np.float32)
        sample['xdir'] = xdir

        return sample




class MultiHuman36M(Dataset):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.labels = np.load(cfg['labels_path'], allow_pickle=True).item()
        # n_cameras = len(self.labels['camera_names'])

        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_subjects = ['S9', 'S11']
        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)
        test_subjects  = list(self.labels['subject_names'].index(x) for x in test_subjects)

        if is_train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
        else:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)


        self.labels['table'] = self.labels['table'][mask]

        self.augment = Compose([
            Crop(cfg['scaleRange'],cfg['moveRange']),
            Resize(cfg['image_size']),
            PhotometricDistort(),
            NormSkeleton(),
            NormImage(),
            GenHeatmap(cfg['num_keypoints'],cfg['image_size'],cfg['heatmap_size']),
        ])

    def __len__(self):
        if self.is_train:
            return len(self.labels['table']) // self.cfg['data_skip_train']
        else:
            return len(self.labels['table']) // self.cfg['data_skip_test']
    
    def __getitem__(self, index):
        if self.is_train:
            index = index * self.cfg['data_skip_train']  + np.random.randint(self.cfg['data_skip_train'])
        else:
            index = index * self.cfg['data_skip_test']  + np.random.randint(self.cfg['data_skip_test'])
        
        shot = self.labels['table'][index]
        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']

        view_list = []
        for camera_idx, camera_name in enumerate(self.labels['camera_names']):

            image = cv2.imread(os.path.join(
            self.cfg['root_path'], subject, action, 'imageSequence' + '-undistorted',
            camera_name, 'img_%06d.jpg' % (frame_idx+1)))

            box = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB

            # x3d = np.pad(
            #     shot['keypoints'][:self.cfg['num_keypts']],
            #     ((0,0), (0,1)), 'constant', constant_values=1.0) 
            x3d = np.asarray(shot['keypoints'][:self.cfg['num_keypoints']])

            shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
            camera = Camera(shot_camera['R'],shot_camera['t'],shot_camera['K'])

            view_list.append(self.augment({'image':image, 'box': box, 'x3d': x3d, 'camera': camera}))

        # random.shuffle(view_list)
        sample = defaultdict(list)
        for view in view_list:
            for key in view:
                sample[key].append(view[key])
        
        for key in sample:
            sample[key] = np.asarray(sample[key])

        return sample