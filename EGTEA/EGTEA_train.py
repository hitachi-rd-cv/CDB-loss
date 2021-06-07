"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import resnext as resnext
from data import *
import torch.optim
import numpy as np
import logging
import argparse
import importlib
from utils.vidaug import augmentors as va
from losses import *

device = torch.device('cuda:0')



def train(model, data_loaders, args):
        
       
    count = np.array([429,779,1421,822,129,1335,130,308,223,249,54,54,139,42,78,117,26,15,22])   #### pre-calculated class frequencies
    freq_ratios = count/count.sum()
     
    best_accuracy = 0.0

    ## prepare optimizer and scheduler

    optimiser = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,  weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=0.6 * args.max_epochs, gamma=0.1)
    
    ##  prepared
    
    ## define loss function
   
    if args.loss_type == 'CE':
        loss = nn.CrossEntropyLoss().to(device)
    elif args.loss_type == 'EQL':
        loss = EQLloss(freq_ratios).to(device)
    elif args.loss_type == 'FL':
        loss = FocalLoss(gamma=args.gamma).to(device)
    elif args.loss_type == 'CB_Softmax':
        loss = CB_Softmax(count, beta=args.beta)
    elif args.loss_type == 'CB_Focal':
        loss = CB_Focal(count, beta=args.beta, gamma=args.gamma)
    elif args.loss_type == 'CDB-CE':
        loss = CDB_loss(class_difficulty=np.ones(args.class_num), tau=args.tau)
    else:
        raise NotImplementedError  
    
    train_loader = data_loaders['train']
 
    best_top_1 = 0.0
    
    for epoch in range(args.max_epochs):
        
        epoch_loss = 0.0
        
            
        correct = 0.0
        model.train()
        for batch_i, (data,targets) in enumerate(train_loader):
            
            data = Variable(data.type(torch.FloatTensor).to(device))
            targets = Variable(targets.type(torch.LongTensor).to(device))
            
            optimiser.zero_grad()

            ## forward prop

            outputs = model(data)
            train_loss = loss(outputs, targets)
            train_loss = torch.mean(train_loss)
            
            ## backprop and update
            
            train_loss.backward()
            optimiser.step()

          
            epoch_loss += train_loss.item()/len(train_loader)
            result = outputs.argmax(1).type(torch.cuda.LongTensor)
            correct += (result == targets).cpu().float().mean()/len(train_loader)
            if batch_i % 50 == 0:
                 print("epoch %d batch/ total_batches: %d/%d  batch train loss : %.5f "%(epoch, batch_i, len(train_loader), train_loss.item()))
        
        scheduler.step()    
        print("Epoch %d average epoch loss %.5f train accuracy %.5f"%(epoch, epoch_loss, correct))
        
        if epoch % args.validate_after_every == 0:
            cls_acc, top_1, prc, rcl = validate(model, data_loaders, args)
            print("Validation ---> top-1 accuracy: %.4f, average precision: %.4f, average recall: %.4f" % (top_1, prc, rcl))
            if args.loss_type == 'CDB-CE':
                 loss = CDB_loss(class_difficulty = 1 - cls_acc, tau=args.tau)
            if top_1 > best_top_1:
                 best_top_1 = top_1
                 torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_model, 'best_model.pth'))

def validate(model, data_loaders, args):
        
        verb_wise_frequency = np.zeros(args.class_num)
        verb_wise_correct = np.zeros(args.class_num)
        verb_wise_predictions = np.zeros(args.class_num)
        model.eval()
        val_loader = data_loaders['val']

        for batch_j, (data, targets) in enumerate(val_loader):
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
            
        ## calculating class accuracies
                   
        class_wise_accuracy = verb_wise_correct/verb_wise_frequency
            
        ## calculating top 1 accuracy

        top_1_accuracy = verb_wise_correct.sum()/verb_wise_frequency.sum()
           
        ## calculating average recall
            
        recall = class_wise_accuracy.mean()

        ## calculating average precision
            
        class_wise_precision = verb_wise_correct/verb_wise_predictions
        class_wise_precision[verb_wise_predictions == 0] = 0
        precision = class_wise_precision.mean()
            
        return class_wise_accuracy, top_1_accuracy, precision, recall   
            
            
            
def main():
    parser = argparse.ArgumentParser(description='Input parameters for EGTEA training')
    parser.add_argument('--train_file', type=str, default='./data/train_split.txt', help='file name for training split')
    parser.add_argument('--val_file', type=str, default='./data/val_split.txt', help='file name for validation split')
    parser.add_argument('--data_root', type=str, help='path to data')
    parser.add_argument('--loss_type', type=str, default='CDB-CE', help='[CE (for CrossEntropy), EQL (for Equalization loss), FL (for focal loss), CB_Softmax(for class-balanced softmax loss), CB_Focal (for class-balanced focal), CDB-CE]')
    parser.add_argument('--tau', type=str, default='dynamic', help = '[0.5, 1, 1.5, 2, 5, dynamic] for CDB-CE loss')
    parser.add_argument('--beta', type=float, default='0.9', help = 'beta value for class-balanced loss')
    parser.add_argument('--gamma', type=float, default=1, help='needed only if you use focal loss and cb-focal loss')
    parser.add_argument('--save_model', type=str, default='./saved_models/')
    parser.add_argument('--validate_after_every', type=int, default=1, help='validate after every n epochs')
    parser.add_argument('--n_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--normalize', type=bool, default=True, help='whether to normalize the weights')
    parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size for 1 gpu')
    parser.add_argument('--frames_per_clip', type=int, default=10, help='number of frames to samples per video clip')
    parser.add_argument('--pretrained_model', type=str, default='./pretrained_weights/resnext-101-kinetics.pth', help='pretrained model path')
    
    args = parser.parse_args()
    
    args.class_num = 19 ## for 19 verb classes

    ## preparing model

    model = resnext.resnet101(shortcut_type='B',cardinality=32, sample_size=224, sample_duration=args.frames_per_clip)
    model = nn.DataParallel(model, device_ids=range(args.n_gpus))
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'])
    model.module.fc = nn.Linear(in_features=2048, out_features=args.class_num, bias=True)
    model.to(device)
    

    ## model prepared
    
    ## create dataloaders

    sometimes = lambda aug: va.Sometimes(0.5, aug)  #for horizontal flipping with prob 0.5
    train_transforms = va.Sequential([va.RandomCrop(size = (480, 480)), va.RandomRotate(degrees=10), sometimes(va.HorizontalFlip()), va.GroupResize(size=(224,224))])  ## data augmentation + pre-processing 
    test_transforms = va.Sequential([va.CenterCrop(size=(480, 480)), va.GroupResize(size=(224,224))])
    train_dataset = createEGTEA(args.train_file, data_root = args.data_root, n_frames_per_clip = args.frames_per_clip, transforms=train_transforms, is_training=True)
    val_dataset = createEGTEA(args.val_file, data_root = args.data_root, n_frames_per_clip = args.frames_per_clip, transforms=test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size * args.n_gpus , shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
    data_loaders = {'train': train_loader, 'val': val_loader}

    ## dataloaders created

    ## start training   
  
    train(model, data_loaders, args)
  
    print('Training Finshed')


if __name__ == '__main__':
    main()

