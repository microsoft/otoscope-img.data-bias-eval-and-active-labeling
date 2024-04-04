# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torchvision.transforms as transforms
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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

def evaluate_model_binary(model,data_loader,criterion,num_class,device):
    model.eval()
    confusion_matrix = torch.zeros(num_class,num_class)
    correct,total = 0,0
    running_loss = 0.0
    prob_arr=np.array([])
    predicted_arr=np.array([])
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            outputs= model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.sum()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return [-running_loss.item()/total,correct / total]


def train_model_binary(model, optimizer, criterion, experiment_folder,train_loader, val_loader, test_loader0,test_loader1=None,test_loader2=None,num_epoch=100, num_class=2,device='cpu'):
    if not os.path.exists(experiment_folder): os.mkdir(experiment_folder)
    train_loss,val_loss,test_loss0,test_loss1,test_loss2=[],[],[],[],[]
    best_metrics=[]
    metric_names=['neg_loss','acc']
    highest_metrics=[-100000000]+[0]
    for epoch in range(num_epoch): 
        # model training
        model.train()
        running_loss = 0.0
        correct=0
        total_num=0
        for i,data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs  = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.sum()
            correct += (predicted == labels).sum().item()
            total_num+=labels.size(0)
        train_loss.append([-running_loss.item()/total_num,correct / total_num])
        current_val_loss=evaluate_model_binary(model,val_loader,criterion,num_class,device)
        val_loss.append(current_val_loss)  
        test_loss0.append(evaluate_model_binary(model,test_loader0,criterion,num_class,device))  
        if test_loader1 is not None: 
            test_loss1.append(evaluate_model_binary(model,test_loader1,criterion,num_class,device))
        if test_loader2 is not None: 
            test_loss2.append(evaluate_model_binary(model,test_loader2,criterion,num_class,device))
        metrics_val=current_val_loss
        for ii in range(len(metrics_val)):
            if metrics_val[ii]>highest_metrics[ii]:
                
                highest_metrics[ii]=metrics_val[ii]
                torch.save(model.state_dict(), os.path.join(experiment_folder,metric_names[ii]+'_model.pt'))
        if epoch%10==9: 
            pd.DataFrame(train_loss).to_csv(os.path.join(experiment_folder,'train_loss.csv'),index=False)
            pd.DataFrame(val_loss).to_csv(os.path.join(experiment_folder,'val_loss.csv'),index=False)
            pd.DataFrame(test_loss0).to_csv(os.path.join(experiment_folder,'test_loss0.csv'),index=False)
            if test_loader1 is not None: pd.DataFrame(test_loss1).to_csv(os.path.join(experiment_folder,'test_loss1.csv'),index=False)
            if test_loader2 is not None: pd.DataFrame(test_loss2).to_csv(os.path.join(experiment_folder,'test_loss2.csv'),index=False)
            torch.save(model.state_dict(), os.path.join(experiment_folder,'model'+str(epoch)+'.pt'))
    pd.DataFrame(train_loss).to_csv(os.path.join(experiment_folder,'train_loss.csv'),index=False)
    pd.DataFrame(val_loss).to_csv(os.path.join(experiment_folder,'val_loss.csv'),index=False)
    pd.DataFrame(test_loss0).to_csv(os.path.join(experiment_folder,'test_loss0.csv'),index=False)
    if test_loader1 is not None: pd.DataFrame(test_loss1).to_csv(os.path.join(experiment_folder,'test_loss1.csv'),index=False)
    if test_loader2 is not None: pd.DataFrame(test_loss2).to_csv(os.path.join(experiment_folder,'test_loss2.csv'),index=False)
    torch.save(model.state_dict(), os.path.join(experiment_folder,'model.pt'))