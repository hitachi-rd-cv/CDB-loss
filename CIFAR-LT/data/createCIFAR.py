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
from PIL import Image
from torch.utils.data import Dataset


class createCIFAR(Dataset):
    def __init__(self, mat, labels, transforms):
        self.mat = mat
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, item):
        image = self.mat[item % len(self.labels)]
        label = self.labels[item % len(self.labels)]
        #image = np.reshape(image, (224, 224))
        image = np.transpose(image,(1,2,0))
        image = Image.fromarray(np.uint8(image))
        image = self.transforms(image).float()
        return image, label
    def __len__(self):
        return len(self.labels)
    def _get_label(self, item):
        return self.labels[item % len(self.labels)]
