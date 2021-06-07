"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the CDB-loss/EGTEA directory.
=======

"""

import numpy as np
import cv2
import os
from joblib import Parallel, delayed
import shutil


def extract_frames(video_out_folder=None, video_path=None):
  if '.mp4' in video_path:
    
    os.makedirs(video_out_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
           
           if frame_no % sample_every == 0:
             cv2.imwrite(os.path.join(video_out_folder, str(int(frame_no/sample_every)).zfill(6)+'.jpg'), frame)
           frame_no += 1
        else:
               cap.release()
               break
        #print(video)

def extract_videowise_frames(vid):
    video_folder = os.path.join(video_dir, vid)
    video_out_folder = os.path.join(home, 'extracted_frames', vid)
    print(video_folder)
    videos = os.listdir(video_folder)
    
    valid_videos = []
    for video in videos:
      if '.mp4' in video:
          valid_videos.append(video)
    y = int(len(valid_videos)/10)
    for i in range(y+1):
       Parallel(n_jobs=10)(delayed(extract_frames)(video_out_folder=os.path.join(video_out_folder, video[:-4]), video_path=os.path.join(video_folder,video)) for video in valid_videos[10 * i: min(len(valid_videos), 10 * (i+1))])

home = '~/datasets/EGTEA/'  #folder where dataset was downloaded. 
video_dir = os.path.join(home, 'cropped_clips') #cropped_clips containes trimmed clips
sample_every = 1   #sampling frequency. assign 6 to sample at 5 fps..
videos = os.listdir(video_dir)
Parallel(n_jobs=len(videos))(delayed(extract_videowise_frames)(videos[i]) for i in range(len(videos)))
           
