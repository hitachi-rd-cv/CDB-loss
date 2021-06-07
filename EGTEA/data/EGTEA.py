"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
<<<<<<< HEAD
LICENSE file in the CDB-loss/EGTEA directory.
=======
LICENSE file in the root directory of this source tree.
>>>>>>> 5fb202a7b9ead5c69fcf597f4f858196677b05f5
"""

import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class createEGTEA(Dataset):
    
    def __init__(self, file_name, data_root='./', image_size=256, n_frames_per_clip=10, transforms=None, is_training=True):

        with open(file_name, 'r') as f:
            self.clip_infos = f.readlines()
        self.data_root = data_root
        self.transforms = transforms
        self.reshape_size = (image_size, image_size)
        self.nf = n_frames_per_clip
        self.is_training = is_training
        self.labels = [int(i.strip('\n').split(' ')[2]) - 1 for i in self.clip_infos]
    
    def __getitem__(self, item):

        single_clip_info = self.clip_infos[item % len(self.clip_infos)]
        
        ##extracting trimmed clip name
        clip_name = single_clip_info.strip('\n').split(' ')[0]
     
        ## extracting raw video name
        video_name = '-'.join(clip_name.split('-')[0:3])
        
        clip_folder = os.path.join(self.data_root, video_name, clip_name)
                
        num_frames = len(os.listdir(clip_folder)) ## number of frames in clip
        
        
        frames = []
        segment_boundaries = np.linspace(start=0, stop=num_frames - 1, num=self.nf + 1)
        
        for i in range(self.nf):
             
             if self.is_training:
                   frame_id = random.sample(range(int(segment_boundaries[i]), int(segment_boundaries[i+1])),1)[0]
             else:
                   frame_id = (segment_boundaries[i] + segment_boundaries[i+1])//2
             
             frame = cv2.imread(os.path.join(clip_folder, str(frame_id).zfill(6) + '.jpg'))
             
             frame.astype(float)
             
             frames.append(frame)
        
        if self.transforms:
             images = self.transforms(frames)
        
        images = np.array(images)
        
        input_img = torch.from_numpy(images).float()
        
        input_img = input_img.permute(3, 0, 1, 2)
    
        label = self.labels[item % len(self.clip_infos)]
        
        
        return input_img, label

    def __len__(self):
 
        return len(self.clip_infos)
