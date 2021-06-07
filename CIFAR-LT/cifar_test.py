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
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import random
import pickle
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from models import *
from data import *



def test(model, dataloaders):
   
    testloader = dataloaders['test']
    model.eval()
    test_total = 0
    test_correct = 0
    for test_images, test_labels in testloader:
          test_images = test_images.cuda()
          test_labels = test_labels.type(torch.cuda.LongTensor)
          with torch.no_grad():
               out = model(test_images)
          _, predicted = out.max(1)
          test_total += test_labels.size(0)
          test_correct += predicted.eq(test_labels).sum().item()
    
    print('Test Accuracy is %.4f'%(test_correct/ test_total)) 
    


def main():
   parser = argparse.ArgumentParser(description='Input parameters for CIFAR-test set testing')
   parser.add_argument('--saved_model_path', type=str, default='./saved_model/best_cifar100_imbalance200.pth', help='model path for testing')
   parser.add_argument('--class_num', type=int, default=100, help='number of classes')
   parser.add_argument('--n_gpus', type=int, default=4, help='number of gpus to use')
   args = parser.parse_args()
   model = resnet.resnet32()
   model.linear = nn.Linear(in_features=64, out_features=args.class_num, bias=True)
   model = nn.DataParallel(model, device_ids=range(args.n_gpus))
   model = model.cuda()
   model.load_state_dict(torch.load(args.saved_model_path))
   
   ## Load test data and labels
   
   test_images, test_labels = load_test_data(args)
   
   ##Loading done
             
   test_images = test_images.reshape(-1, 3, 32, 32)  ## reshape to N, C, H, W
   transform_test = transforms.Compose([
       transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])
   test_set = createCIFAR(test_images, test_labels, transforms=transform_test)
   testloader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)
   dataloaders = {'test': testloader}
   test(model, dataloaders)

if __name__ == '__main__':
   main()

   
