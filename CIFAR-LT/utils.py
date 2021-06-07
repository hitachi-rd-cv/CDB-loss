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




def sigmoid(x):
   return (1/(1+np.exp(-x)))

def compute_weights(class_wise_accuracy, tau='dynamic', normalize=True, epsilon=0.01):
    if tau == 'dynamic':
        bias = np.max(class_wise_accuracy)/(np.min(class_wise_accuracy) + epsilon)
        tau = 2 * sigmoid(bias-1)
    else:
        tau = float(tau)
    cdb_weights = (1 - class_wise_accuracy)**tau
    assert (cdb_weights < 0).sum() == 0
    if normalize:
         cdb_weights = cdb_weights/ cdb_weights.sum() * len(cdb_weights)   #normalizing weights to make sum of weights = number of classes
    
    return torch.FloatTensor(cdb_weights).cuda()
