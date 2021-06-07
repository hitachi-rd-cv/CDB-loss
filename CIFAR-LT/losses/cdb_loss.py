"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the CDB-loss/CIFAR-LT directory.
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