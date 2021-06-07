"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the CDB-loss/EGTEA directory.
=======
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import *
from data import *
import numpy as np
from utils.vidaug import augmentors as va


device = torch.device('cuda:0')
count = np.array([429,779,1421,822,129,1335,130,308,223,249,54,54,139,42,78,117,26,15,22]) ## count of verbs in train data


def test(model, data_loaders, args):
    
    test_loader = data_loaders['test']

    verb_wise_frequency = np.zeros(args.class_num)   ### for calculating frequency of each verb in the test set
    verb_wise_correct = np.zeros(args.class_num)     ### for calculating correct classifications per class
    verb_wise_predictions = np.zeros(args.class_num) ### for calculating predictions per class

    model.eval()

    for batch_j, (data, targets) in enumerate(test_loader):
         data = Variable(data.type(torch.FloatTensor).to(device))
         targets = Variable(targets.type(torch.LongTensor).to(device))
         with torch.no_grad():
              outputs = model(data)
         
         results = outputs.argmax(1).type(torch.cuda.LongTensor)
         correct = targets[targets == results]
         for cls in range(args.class_num):
              verb_wise_frequency[cls] += (targets == cls).sum().item()
              verb_wise_predictions[cls] += (results == cls).sum().item()
              verb_wise_correct[cls] += (correct == cls).sum().item()
    
    ## calculating top-1 accuracy
  
    top_1 = verb_wise_correct.sum() / verb_wise_frequency.sum()

    ## calculating average recall

    class_wise_recall = verb_wise_correct / verb_wise_frequency
    recall = class_wise_recall.mean()  

    ## calculating average precision

    class_wise_precision = verb_wise_correct / verb_wise_predictions
    class_wise_precision[verb_wise_predictions == 0] = 0
    precision = class_wise_precision.mean()
                
    ## calculating average precision and recall for majority classes
        
    maj_class_prec = class_wise_precision[count>600].mean()
    maj_class_recall = class_wise_recall[count>600].mean()

    ## calculating average precision and recall for minority classes

    min_class_prec = class_wise_precision[count<600].mean()
    min_class_recall = class_wise_recall[count<600].mean()
    
    print('Testing -----> top-1 accuracy: %.4f, average precision: %.4f, average recall: %.4f' % (top_1, precision, recall))
    print('Testing -----> Majority class: Precision - %.5f Recall - %.5f , Minority class: Precision - %.5f Recall - %.5f' % (maj_class_prec, maj_class_recall, min_class_prec, min_class_recall))


def main():
    parser = argparse.ArgumentParser(description='Input parameters for EGTEA training')
    parser.add_argument('--test_file', type=str, default='./data/test_split.txt', help='file name for testing split')
    parser.add_argument('--data_root', type=str, help='path to data')
    parser.add_argument('--trained_model', type=str, default='./saved_models/best_model.pth', help='path to trained model')
    parser.add_argument('--frames_per_clip', type=int, default=10, help='number of frames to sample from each clip')
    parser.add_argument('--n_gpus', type=int, default=4, help='number of gpus')
  
    args= parser.parse_args()
 
    args.class_num = 19 ## 19 classes for EGTEA

    ## preparing model

    model = resnext.resnet101(shortcut_type='B',cardinality=32, sample_size=224, sample_duration=args.frames_per_clip)
    model = nn.DataParallel(model, device_ids=range(args.n_gpus))
    model.module.fc = nn.Linear(in_features=2048, out_features=args.class_num, bias=True)
    model.load_state_dict(torch.load(args.trained_model)['state_dict'])
    model.to(device)

    ## model prepared
    
    ## creating dataloader
   
    test_transforms = va.Sequential([va.CenterCrop(size=(480, 480)), va.GroupResize(size=(224,224))])
    test_dataset = createEGTEA(args.test_file, data_root = args.data_root, n_frames_per_clip = args.frames_per_clip, transforms=test_transforms)
    data_loaders = {'test': DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)}
    
    ## dataloader created

    test(model, data_loaders, args)


if __name__ == '__main__':
    main()

