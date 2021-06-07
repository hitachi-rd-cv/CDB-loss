"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the CDB-loss/CIFAR-LT directory.
=======
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class EQLloss(nn.Module):
    def __init__(self, freq_info):
        super(EQLloss, self).__init__()
        self.freq_info = freq_info
        # self.pred_class_logits = pred_class_logits
        # self.gt_classes = gt_classes
        self.lambda_ = 0.03
        self.gamma = 0.95
    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def forward(self, pred_class_logits, gt_classes,):
        self.pred_class_logits = pred_class_logits
        self.gt_classes = gt_classes
        self.n_i, self.n_c = self.pred_class_logits.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(self.pred_class_logits, self.gt_classes)
        if torch.rand(1).item() > self.gamma:
            coeff = torch.zeros(1)
        else:
            coeff = torch.ones(1)
        coeff = coeff.cuda()
        eql_w = 1 - (coeff * self.threshold_func() * (1 - target))

        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, target,
                                                      reduction='none')

        return torch.sum(cls_loss * eql_w) / self.n_i
