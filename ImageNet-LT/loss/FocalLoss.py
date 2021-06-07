
"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

""" 



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        ones = - torch.ones(input.shape).cuda()
        for i in range(input.shape[0]):
            ones[i, target[i,0].item()] = 1 
        input = input * ones
        #logpt = F.log_softmax(input)
        logpt = F.logsigmoid(input)
        #logpt = logpt.gather(1,target)
        #logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        #print(pt)
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.sum(1)
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            #logpt = logpt * Variable(at)
            loss = loss * Variable(at)

        #loss = -1 * (1-pt)**self.gamma * logpt
        if reduction == 'mean': return loss.mean()
        elif reduction == 'sum': return loss.sum()
        else: return loss