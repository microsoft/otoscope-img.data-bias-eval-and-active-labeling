# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
sys.path.append('../prepare_dataset')
sys.path.append('../models')
import os
import pdb 
import argparse
import pandas as pd
import torch.optim as optim
import prepare_binary_dataset
import data_aug
import models_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import train_model_binary
import time


def write_file(filename, **kwargs):
    with open(filename, "w") as handle:
        for key, value in kwargs.items():
            handle.write("{}: {} {} \n" .format(key, value, type(value)))
            
parser = argparse.ArgumentParser()


parser.add_argument( "--model_name", default='vgg16',type=str)
parser.add_argument( "--eclipse", action='store_true',help="eclipse image or not")
parser.add_argument( "--eclipse_extent", default=0.0,type=float,help="eclipse extent ranging from 0 to 1")
parser.add_argument( "--oversample", action='store_true',help="whether to oversample")
parser.add_argument( "--batch_size", default=32,type=int,help="batch size")
parser.add_argument( "--seed", default=1234,type=int,help="random seed for data split")
parser.add_argument( "--num_epoch", default=100,type=int)
parser.add_argument( "--elastic_tf", action='store_true')
parser.add_argument( "--add_gauss_noise", action='store_true')
parser.add_argument( "--colorhue", default=0.05,help="color hue",type=float)
parser.add_argument( "--lr", default=0.01,help="learning rate",type=float)
parser.add_argument( "--wd", default=0,help="weight decay",type=float)
parser.add_argument( "--cudaID", default=0,type=int,help="cuda ID")
parser.add_argument( "--scale", default=0.9,type=float)

args = parser.parse_args()
arg_dict=vars(parser.parse_args())
del arg_dict['cudaID']
arg_list=arg_dict.values()
DATA_DIR='../../../data/eardrum_public_data'
EXP_MAIN_FOLDER='../experiment_gen'
if not os.path.exists(EXP_MAIN_FOLDER): os.mkdir(EXP_MAIN_FOLDER)
save_folder_name='_'.join([str(i) for i in arg_list])
save_dir=os.path.join(EXP_MAIN_FOLDER,save_folder_name)
if not os.path.exists(save_dir): 
    os.mkdir(save_dir)
else: 
    save_dir=save_dir+'_'+str(int(time.time()))
    os.mkdir(save_dir)
    
write_file(os.path.join(save_dir,'parameters.txt'), 
           model_name=args.model_name,
           scale=args.scale,
           lr=args.lr,wd=args.wd,
           batch_size=args.batch_size,
           seed=args.seed,
           num_epoch=args.num_epoch,
           add_gauss_noise=args.add_gauss_noise,
           elastic_tf=args.elastic_tf,
           colorhue=args.colorhue,
           eclipse=args.eclipse,
           eclipse_extent=args.eclipse_extent,
           oversample=args.oversample)


img_df=pd.read_csv('../metadata/metadata.csv')
device=torch.device('cuda:'+str(args.cudaID))

criterion = nn.CrossEntropyLoss(reduction='none')
num_class=2

for index in range(3): 
    origins=['Chile','Ohio','Turkey']
    # train on origins[index] and validate on the others
    experiment_folder=os.path.join(save_dir,origins[index])
    train_origin=origins[index]
    print(train_origin)
    train0=img_df[img_df.source==origins.pop(index)]
    test0=img_df[img_df.source==origins[0]]
    test1=img_df[img_df.source==origins[1]]
    print('test0 test1:',origins[0],origins[1])
    if train_origin=='Ohio':
        train,val=train_test_split(train0,stratify=train0['class'],test_size=0.2,shuffle=True,random_state=args.seed)
    else:
        train=train0[train0.is_test==False].reset_index(drop=True)
        val=train0[train0.is_test==True].reset_index(drop=True)
    model = models_classification.model_classification(args.model_name)
    model.cuda(args.cudaID)
    optimizer = optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.wd)
    train_tf,test_tf=data_aug.derive_transform(model.size,model.mean,model.std,
                                               scale=args.scale,
                     add_gauss_noise=args.add_gauss_noise,
                     elastic_tf=args.elastic_tf,
                     color_hue=args.colorhue)
    
    train_data=prepare_binary_dataset.OtoDataset_Binary(train,transform=train_tf,data_dir=DATA_DIR,
    eclipse=args.eclipse,eclipse_extent=args.eclipse_extent)
    val_data=prepare_binary_dataset.OtoDataset_Binary(val,transform=test_tf,data_dir=DATA_DIR,
    eclipse=args.eclipse,eclipse_extent=args.eclipse_extent)
    test_data0=prepare_binary_dataset.OtoDataset_Binary(test0,transform=test_tf,data_dir=DATA_DIR,
    eclipse=args.eclipse,eclipse_extent=args.eclipse_extent)
    test_data1=prepare_binary_dataset.OtoDataset_Binary(test1,transform=test_tf,data_dir=DATA_DIR,
    eclipse=args.eclipse,eclipse_extent=args.eclipse_extent)
    train_loader=DataLoader(train_data,batch_size=args.batch_size,num_workers=4,shuffle = True)

    val_loader=DataLoader(val_data,batch_size=args.batch_size,num_workers=4)
    test_loader0=DataLoader(test_data0,batch_size=args.batch_size,num_workers=4)
    test_loader1=DataLoader(test_data1,batch_size=args.batch_size,num_workers=4)
    train_model_binary.train_model_binary(model=model, optimizer=optimizer, criterion=criterion,
        train_loader=train_loader, val_loader=val_loader, test_loader0=test_loader0,test_loader1=test_loader1,
        num_epoch=args.num_epoch,experiment_folder=experiment_folder, num_class=num_class,device=device)
