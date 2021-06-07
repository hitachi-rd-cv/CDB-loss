"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
<<<<<<< HEAD
LICENSE file in the CDB-loss/CIFAR-LT directory.
=======
LICENSE file in the root directory of this source tree.
>>>>>>> 5fb202a7b9ead5c69fcf597f4f858196677b05f5
"""

import numpy as np
import torch
import torch.nn as nn


def sigmoid(x):
  return (1/(1+np.exp(-x)))


class CDB_loss(nn.Module):
  
    def __init__(self, class_difficulty, tau='dynamic', reduction='none'):
        
        super(CDB_loss, self).__init__()
        self.class_difficulty = class_difficulty
        if tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/(1 - np.max(class_difficulty) + 0.01)
            tau = sigmoid(bias)
        else:
            tau = float(tau) 
        self.weights = self.class_difficulty ** tau
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weights), reduction=self.reduction).cuda()
        

    def forward(self, input, target):

        return self.loss(input, target)