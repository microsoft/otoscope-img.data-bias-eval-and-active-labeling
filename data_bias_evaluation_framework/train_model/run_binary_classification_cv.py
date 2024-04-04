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
EXP_MAIN_FOLDER='../experiment'
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

device=torch.device('cuda:'+str(args.cudaID))

criterion = nn.CrossEntropyLoss(reduction='none')
num_epoch=args.num_epoch


num_class=2
criterion = nn.CrossEntropyLoss(reduction='none')
img_df=pd.read_csv('../metadata/metadata.csv')

test_fold_id=0

img_df['source_class']=img_df['source']+'_'+img_df['class']
skf = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)

for train_index, test_index in skf.split(img_df, img_df['source_class']): 
    experiment_folder=os.path.join(save_dir,str(test_fold_id))
    train0=img_df.iloc[train_index,]
    test=img_df.iloc[test_index,]
    train,val=train_test_split(train0,stratify=train0['source_class'],test_size=0.2,shuffle=True,random_state=args.seed)
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
    test_data=prepare_binary_dataset.OtoDataset_Binary(test,transform=test_tf,data_dir=DATA_DIR,
    eclipse=args.eclipse,eclipse_extent=args.eclipse_extent)

    if args.oversample:
        train_labels=train['binary_class']
        total_train=train.shape[0]
        class_weights=(total_train/train_labels.value_counts()).to_dict()
        train_weights=[class_weights[int(label)] for label in train_labels]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_labels),replacement=True)
        train_loader=DataLoader(train_data,batch_size=args.batch_size,num_workers=4,sampler=sampler)
    else: 
        train_loader=DataLoader(train_data,batch_size=args.batch_size,num_workers=4,shuffle = True)


    train_loader=DataLoader(train_data,batch_size=args.batch_size,num_workers=4,shuffle = True)
    val_loader=DataLoader(val_data,batch_size=args.batch_size,num_workers=4,shuffle = False)
    test_loader=DataLoader(test_data,batch_size=args.batch_size,num_workers=4,shuffle = False)
    train_model_binary.train_model_binary(model=model, optimizer=optimizer, criterion=criterion,
        train_loader=train_loader, val_loader=val_loader, test_loader0=test_loader,
        num_epoch=num_epoch,experiment_folder=experiment_folder, num_class=num_class,device=device)
    test_fold_id+=1
