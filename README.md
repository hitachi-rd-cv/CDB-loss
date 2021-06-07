# Class-Wise Difficulty-Balanced Loss for Solving Class-Imbalance
____
by Saptarshi Sinha, Hiroki Ohashi and Katsuyuki Nakamura

This repository contains the official implementation of the paper 'Class-Wise Difficulty-Balanced Loss for Solving Class-Imbalance', which was accepted for Oral presentation at ACCV, 2020.  [(paper)] [(arXiv)]

The code has been organized under 3 folders namely 'CIFAR-LT', 'EGTEA' and 'ImageNet-LT' that represents the 3 datasets that we have used in our paper.

### Requirements
___
The environment required to successfully implement our paper mainly needs
```
- Python >= 3.6
- PyTorch == 1.5.0
- Opencv-python == 4.1.2
- Pillow
- PyYaml
```
### Experiments on CIFAR-LT
___
```
cd CDB-loss/CIFAR-LT
```
Please download the [CIFAR-100] data and extract it in `./data/`.

To start training on CIFAR-LT using our CDB-CE loss, 

```
python cifar_train.py --class_num 100 --imbalance 200 --loss_type CDB-CE --tau 1.5 --n_gpus 1
```
Use `-- class_num 100` for  CIFAR100-LT. Select the amount of imbalance you want to inject in the dataset by using `-- imbalance 200`. Note `-- imbalance 1` means no imbalance will be injected. 

To evaluate the best model on the balanced test set,
```
python cifar_test.py --saved_model_path saved_model/best_cifar100_imbalance200.pth --class_num 100 --n_gpus 1
```
Select the appropriate saved model and evaluate. The output reads like
```
Test Accuracy is 0.3740
```

### Experiments on ImageNet-LT
___
For ImageNet-LT, we build our codes on the code from [classifier-balancing]. To reproduce results of [classifier-balancing], please follow [this].

Download the dataset [ImageNet2014]. Accordingly change the `data_root` in `CDB-loss/ImageNet-LT/main.py`.
```
cd CDB-loss/ImageNet-LT
```
To train a ResNet10 on ImageNet-LT using our CDB-CE loss,
```
python main.py --cfg ./config/ImageNet_LT/feat_uniform_with_CDBloss.yaml
```
In the config file, you can change the value of tau for our loss function.
To evaluate the final model on the test set,
```
python main.py --test --model_dir ./logs/ImageNet_LT/models/resnet10_uniform_cdbce
```

### Experiments on EGTEA
___
Download the trimmed action clips and annotations for [EGTEA] dataset. Extract the frames using `CDB-loss/EGTEA/data/extract_frames.py` and save the frames under `extracted_frames`.
        The folder structure should look like this.
```
datasets
|
|
|__EGTEA
    |
    |__  extracted_frames
          |
          |___  OP01-R01-PastaSalad
          |      |
          |      |___  OP01-R01-PastaSalad-1002316-1004005-F024051-F024101     
          |      |              |__ 000000.jpg       
          |      |              |__ 000001.jpg
          |      |              |__  ...
          |      |___  OP01-R01-PastaSalad-1004110-1021110-F024057-F024548
          |      |                   ...
          |                         
          |____  OP01-R02-TurkeySandwich
          |      |
          |      |___   OP01R02-TurkeySandwich-102320-105110-F002449-F002529
          |                                   ...
          |   ...

```
The train/val/test splits used for our experiments are provided under `CDB-loss/EGTEA/data`.
```
cd CDB-loss/EGTEA
```
Download `resnext-101-kinetics.pth` from [here] and save it under `./pretrained_weights/` .

To train a 3D-ResNeXt101 on EGTEA dataset using CDB-CE loss,
```
python EGTEA_train.py --data_root ~/datasets/EGTEA/extracted_frames --loss_type CDB-CE --tau 1.5 --n_gpus 2
```
Provide absolute path to `extracted_frames` as `data_root`.

To evaluate the final model,
```
python EGTEA_test.py --data_root ~/datasets/EGTEA/extracted_frames --trained_model ./models/best_model.pth --n_gpus 2
```

### Citations
____
If you use our code, please consider citing our paper as

```
@InProceedings{Sinha_2020_ACCV,
author={Sinha, Saptarshi and Ohashi, Hiroki and Nakamura, Katsuyuki},
title={Class-Wise Difficulty-Balanced Loss for Solving Class-Imbalance},
booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
month={November},
year={2020}
}
```
For queries, contact at saptarshi.sinha.hx@hitachi.com

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

    
   [classifier-balancing]: <https://github.com/facebookresearch/classifier-balancing> 
   [(paper)]: <https://openaccess.thecvf.com/content/ACCV2020/papers/Sinha_Class-Wise_Difficulty-Balanced_Loss_for_Solving_Class-Imbalance_ACCV_2020_paper.pdf>
   [CIFAR-100]: <https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz>
   [(arXiv)]: <https://arxiv.org/abs/2010.01824>
   [ImageNet2014]: <https://image-net.org/index>
   [EGTEA]: <http://cbs.ic.gatech.edu/fpv/>
   [here]: <https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M>
   [this]: <https://github.com/facebookresearch/classifier-balancing/blob/master/README.md>
 
   
   
   
