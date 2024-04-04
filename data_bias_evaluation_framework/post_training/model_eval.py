# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
sys.path.append('../prepare_dataset')
sys.path.append('../models')
import os
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torchvision
import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import utils
import data_aug
import prepare_binary_dataset
import models_classification
from sklearn.model_selection import train_test_split
def derive_auc(model,data_loader,device,thresh=0.5):
    pred=[]
    y=[]
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            pred+=nn.Softmax(dim=1)(outputs).cpu().numpy()[:,1].tolist()
            y+=labels.cpu().numpy().tolist()
    return metrics.roc_auc_score(y, pred),metrics.accuracy_score(y,[xx>thresh for xx in pred]),metrics.average_precision_score(y, pred)

def summarize_experiment(experiment_main_folder,cudaID,model_state_name='model.pt'):
    df_metrics=pd.DataFrame(columns=['model_name','eclipse','eclipse_extent',
                                     'train_origin',
                                 'val_acc', 'val_auc','val_prauc',
                                 'test0_acc','test0_auc','test0_prauc',
                                 'test1_acc','test1_auc','test1_prauc'])
    model_state_name='model.pt'
    batch_size=32
    device='cuda:'+str(cudaID)
    img_df=pd.read_csv('../metadata/metadata.csv')
    DATA_DIR='../../../data/eardrum_public_data'
    param_dict=utils.read_param_file(os.path.join(experiment_main_folder,'parameters.txt'))
    for index in range(3): 
        origins=['Chile','Ohio','Turkey']
        # train on origins[index] and validate on the others
        experiment_folder=os.path.join(experiment_main_folder,origins[index])
        train_origin=origins[index]
        print(train_origin)
        train0=img_df[img_df.source==origins.pop(index)]
        test0=img_df[img_df.source==origins[0]]
        test1=img_df[img_df.source==origins[1]]
        print('test0 test1:',origins[0],origins[1])
        if train_origin=='Ohio':
            train,val=train_test_split(train0,stratify=train0['class'],test_size=0.2,shuffle=True,random_state=param_dict['seed'])
        else:
            train=train0[train0.is_test==False].reset_index(drop=True)
            val=train0[train0.is_test==True].reset_index(drop=True)
        model = models_classification.model_classification(param_dict['model_name'])
        model.cuda(cudaID)
        model.load_state_dict(torch.load(
            os.path.join(experiment_folder,model_state_name)
        ))
        train_tf,test_tf=data_aug.derive_transform(model.size,model.mean,model.std,
                                                scale=param_dict['scale'],
                        add_gauss_noise=param_dict['add_gauss_noise'],
                        elastic_tf=param_dict['elastic_tf'],
                        color_hue=param_dict['colorhue'])
        val_data=prepare_binary_dataset.OtoDataset_Binary(val,transform=test_tf,data_dir=DATA_DIR,
        eclipse=param_dict['eclipse'],eclipse_extent=param_dict['eclipse_extent'])
        test_data0=prepare_binary_dataset.OtoDataset_Binary(test0,transform=test_tf,data_dir=DATA_DIR,
        eclipse=param_dict['eclipse'],eclipse_extent=param_dict['eclipse_extent'])
        test_data1=prepare_binary_dataset.OtoDataset_Binary(test1,transform=test_tf,data_dir=DATA_DIR,
        eclipse=param_dict['eclipse'],eclipse_extent=param_dict['eclipse_extent'])
        val_loader=DataLoader(val_data,batch_size=batch_size,num_workers=4)
        test_loader0=DataLoader(test_data0,batch_size=batch_size,num_workers=4)
        test_loader1=DataLoader(test_data1,batch_size=batch_size,num_workers=4)
        auc_val,acc_val,prauc_val=derive_auc(model,val_loader,device,thresh=0.5)
        print(auc_val,acc_val,prauc_val)
        auc_test0,acc_test0,prauc_test0=derive_auc(model,test_loader0,device,thresh=0.5)
        print(auc_test0,acc_test0,prauc_test0)
        auc_test1,acc_test1,prauc_test1=derive_auc(model,test_loader1,device,thresh=0.5)
        print(auc_test1,acc_test1,prauc_test1)
        df_metrics.loc[index]=[param_dict['model_name'],param_dict['eclipse'],param_dict['eclipse_extent'],
                                      train_origin,acc_val,auc_val,prauc_val,
                                      acc_test0,auc_test0,prauc_test0,
                                      acc_test1,auc_test1,prauc_test1]
    return df_metrics
