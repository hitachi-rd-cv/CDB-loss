"""Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

"""
import numpy as np
import sys
import torch
import torch.nn as nn
from loss.FocalLoss import FocalLoss

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def create_loss (class_difficulty=None, loss_params=('dynamic', None, True)):
  
    print('Using CDB Softmax Loss')
    tau, focal_gamma, normalize = loss_params
    
    if class_difficulty is not None:
        epsilon = 0.01
        if tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/ (1 - np.max(class_difficulty) + epsilon) - 1
            tau = 2 * sigmoid(bias)
        else:
            tau = float(tau)
        
        cdb_weights = class_difficulty ** tau
        if normalize:
           cdb_weights = (cdb_weights / cdb_weights.sum()) * len(cdb_weights)      
        if focal_gamma is not None:
             return FocalLoss(gamma=float(focal_gamma), alpha=torch.FloatTensor(cdb_weights))
        else:
             return nn.CrossEntropyLoss(weight=torch.FloatTensor(cdb_weights),)
    else:
        sys.exit('Class Difficulty can not be None')