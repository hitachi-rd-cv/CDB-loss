"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn as nn
from .focal_loss import *


class CB_loss(nn.Module):
     
     def __init__(self, beta, samples_per_cls, reduction='none'):
         
         super(CB_loss, self).__init__()
         self.beta = beta
         self.samples_per_cls = samples_per_cls
         self.reduction = reduction
         self.device = torch.device('cuda:0')


     def compute_weights(self,):
   
         effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
         weights = (1.0 - self.beta) / np.array(effective_num)
         weights = weights / np.sum(weights) * len(self.samples_per_cls)
         self.weights = torch.Tensor(weights)

   
     def forward(self, input, target):
         
         raise NotImplementedError
      


class CB_Softmax(CB_loss):
     
     def __init__(self, samples_per_cls, beta=0.9, reduction='none'):
          
           super().__init__(beta, samples_per_cls, reduction)
           self.compute_weights()
           self.loss = nn.CrossEntropyLoss(weight=self.weights, reduction=self.reduction).to(self.device)

     
     def forward(self, input, target):
           
          return self.loss(input, target)
 

class CB_Focal(CB_loss):

      def __init__(self, samples_per_cls, beta=0.9, gamma=1, reduction='none'):

           super().__init__(beta, samples_per_cls, reduction)
           self.compute_weights()
           self.loss = FocalLoss(alpha=self.weights, gamma=gamma, reduction=self.reduction).to(self.device)

     
      def forward(self, input, target):
          
          return self.loss(input, target)
