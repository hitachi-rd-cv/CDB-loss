"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the CDB-loss/CIFAR-LT directory.
=======

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import argparse
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from models import *
from losses import *
from data import *
from utils import *




def train(model, dataloaders, args):
   
   params_dict = dict(model.named_parameters())
   params = []
   for key, value in params_dict.items():
        if key == 'module.linear.bias':
            params += [{'params':value, 'weight_decay':0}]
        else:
            params += [{'params':value, 'weight_decay':5e-4}]

   ## define optimizer and scheduler
   
   optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
   scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 210], gamma=0.01)
   
   ## create loss
   
   if args.loss_type == 'CE':
       loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
   elif args.loss_type == 'FL':
       loss = FocalLoss(gamma=args.gamma, size_average=False).cuda()
   elif args.loss_type == 'EQL':
       loss = EQLloss(args.freq_ratio, gamma=args.eql_gamma, lamda=args.eql_lambda).cuda()
   elif args.loss_type == 'CDB-CE':
       loss = CDB_loss(class_difficulty = np.ones(args.class_num), tau=args.tau).cuda()
   else :
       sys.exit('Sorry. No such loss function implemented')
   ## create folder for saving model
      
   os.makedirs(args.save_model, exist_ok = True)
   
   trainloader = dataloaders['train']

   best_val_accuracy = 0

   for epoch in range(args.max_epochs):
      epoch_loss = 0
      total_train_samples = 0
      model.train()
      
      for batch_i, (train_images, train_labels) in enumerate(trainloader):
         train_images = Variable(train_images.cuda(), requires_grad=False)
         train_labels = Variable(train_labels.type(torch.cuda.LongTensor), requires_grad=False)
         optimizer.zero_grad()
         out = model(train_images)
         train_loss = loss(out, train_labels)
         epoch_loss += torch.sum(train_loss).item()
         total_train_samples += len(train_loss)
         train_loss = torch.mean(train_loss)
         train_loss.backward()
         optimizer.step()
      print('Epoch %d/%d train loss = %.5f' % (epoch+1, args.max_epochs, epoch_loss/total_train_samples))
      scheduler.step()
      if epoch % args.validate_after_every == 0:
         class_wise_accuracy = validate(model, dataloaders, args)
         if args.loss_type == 'CDB-CE':
            #cdb_weights = compute_weights(class_wise_accuracy, tau = args.tau, normalize = args.normalize)
            loss = CDB_loss(class_difficulty = 1 - class_wise_accuracy, tau = args.tau).cuda()
         val_accuracy = class_wise_accuracy.mean()
         print('Validation: val accuracy = %.4f' % (val_accuracy))
         if val_accuracy > best_val_accuracy:
             best_val_accuracy = val_accuracy
             torch.save(model.state_dict(), os.path.join(args.save_model, 'best_cifar{}_imbalance{}.pth'.format(args.class_num, args.imbalance)))

def validate(model, dataloaders, args):
     model.eval()
     val_total = 0
     val_loss = 0
     #val_correct = 0
     class_wise_accuracy = np.zeros(args.class_num)
     #validation_loss = nn.CrossEntropyLoss(reduction='none').cuda()
     valloader = dataloaders['val']
     for val_images, val_labels in valloader:
          val_images = val_images.cuda()
          val_labels = val_labels.type(torch.cuda.LongTensor)
          with torch.no_grad():
               out = model(val_images)
          _, val_predicted = out.max(1)
          #val_loss += validation_loss(out, val_labels).sum().item()
          val_total += val_labels.size(0)
          #val_correct += val_predicted.eq(val_labels).sum().item()
          for id in range(len(val_predicted)): 
             if val_predicted[id] == val_labels[id]:
                 class_wise_accuracy[int(val_predicted[id])] += 1
     
     class_wise_accuracy = class_wise_accuracy/args.val_samples_per_class
     return class_wise_accuracy


def main():
    parser = argparse.ArgumentParser(description='Input parameters for CIFAR-LT training')
    parser.add_argument('--class_num', type=int, default=100, help='number of classes (100 for CIFAR100)')
    parser.add_argument('--imbalance', type=int, default=200, help='imbalance ratios in [200, 100, 50, 20, 10, 1(no imbalance)]')
    parser.add_argument('--loss_type', type=str, default='CDB-CE', help='[CE (CrossEntropy), EQL (EqualizationLoss), FL (FocalLoss), CDB-CE (Ours)]')
    parser.add_argument('--tau', type=str, default='dynamic', help='[0.5, 1, 1.5, 2, 5, dynamic]')
    parser.add_argument('--gamma', type=float, default=1, help='only if you use focal loss')
    parser.add_argument('--eql_gamma', type=float, default=0.9, help='equalization loss gamma')
    parser.add_argument('--eql_lambda', type=float, default=0.005, help='equalization loss lambda')
    parser.add_argument('--save_model', type=str, default='./saved_model/')
    parser.add_argument('--validate_after_every', type=int, default=1, help='validate after every n epochs')
    parser.add_argument('--n_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--normalize', type=bool, default=True, help='whether to normalise the weights')
    parser.add_argument('--max_epochs', type=int, default=240, help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for 1 gpu during training')
    args = parser.parse_args()
    
    ###prepare ResNet32
   
    model = resnet.resnet32()
    model.linear = nn.Linear(in_features=64, out_features=args.class_num, bias=True)
    pi = 1/args.class_num
    b = -np.log((1-pi)/pi)
    model.linear.bias.data.fill_(b)
    model = nn.DataParallel(model, device_ids=range(args.n_gpus))  #for using multiple gpus
    model = model.cuda()
    print('model prepared')

    ###preparation of model done

    ###preparation of data

    ## loading all train images and labels
    
    train_images, train_labels = load_data(args) 
    
    ## loading done
    
    
    
    ## separation of all train data into sub-train and val 

    args.val_samples_per_class = 25     
    sub_train_images, sub_train_labels, val_images, val_labels = sep_train_val(train_images, train_labels, args)
    
    ## separation done
    
    ## creating imbalance in the dataset
    
    imbalanced_train_images, imbalanced_train_labels = create_imbalance(sub_train_images, sub_train_labels, args)
    
    ##imbalance created
    
    imbalanced_train_images = imbalanced_train_images.reshape(-1, 3, 32, 32)  ##reshape to N * C * H * W
    val_images = val_images.reshape(-1, 3, 32, 32)

    _, class_wise_freq = np.unique(imbalanced_train_labels, return_counts=True)
    args.freq_ratio = class_wise_freq / class_wise_freq.sum()
    
    ###preparation of data done

    ###creation of data loaders for train and val

    img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), img_normalize])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    img_normalize,
     ])
    train_dataset = createCIFAR(imbalanced_train_images, imbalanced_train_labels, transforms=transform_train)
    val_dataset = createCIFAR(val_images, val_labels, transforms=transform_test)
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size * args.n_gpus, num_workers=8)
    valloader = DataLoader(val_dataset, shuffle=False, batch_size=256, num_workers=8)
    dataloaders = {'train': trainloader, 'val': valloader}

    ### dataloader creation finished

    train(model, dataloaders, args)
    print('Training Finished')


if __name__ == '__main__':
    main()
          