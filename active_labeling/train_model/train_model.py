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
import math



def derive_multiclassification_metircs(confusion_matrix):
    num_class=confusion_matrix.shape[0]
    recalls=[]
    precisions=[]
    for class_index in range(num_class):
        if confusion_matrix.sum(axis=1)[class_index]!=0: 
        # If there is at least one example belonging to class class_index. 
        #Calculate its recall and precision. Ow skip this class. 
            re=(confusion_matrix[class_index,class_index]/confusion_matrix.sum(axis=1)[class_index]).item()
            pr=(confusion_matrix[class_index,class_index]/confusion_matrix.sum(axis=0)[class_index]).item() 
            recalls.append(re)
            precisions.append(pr)
    acc=(torch.diagonal(confusion_matrix).sum()/confusion_matrix.sum()).item()
    recall_macroavg=sum(recalls)/len(recalls)
    precision_macroavg=sum(precisions)/len(precisions)
    F1_macro=2*precision_macroavg*recall_macroavg/(recall_macroavg+precision_macroavg)
    return [acc,recall_macroavg,F1_macro,precision_macroavg] 

def evaluate_model_multiclass(model,data_loader,criterion,num_class,device,print_acc=False,model_multihead=False):
    model.eval()
    confusion_matrix = torch.zeros(num_class,num_class)
    confusion_matrix_c = torch.zeros(num_class,num_class)
    correct,correct_c,total = 0,0,0
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            images, label1,label2,label3 = data
            labels=label3
            images=images.to(device)
            labels=labels.to(device)
            if model_multihead: _,_,outputs = model(images)
            else: outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.sum()
            _, predicted = torch.max(outputs, 1)
            predicted_probs=nn.Softmax(dim=1)(outputs)   
            total += labels.size(0)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    metric_list=[-running_loss.item()/total]+derive_multiclassification_metircs(confusion_matrix)
    if print_acc:
        print('neg_loss,acc,recall_avg,F1',metric_list)
    return metric_list






def train_model_multitask(
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        test_loader,
        experiment_folder,
        num_epoch, 
        num_class,
        alpha_level2=1,
        alpha_level1=1,
        device='cpu',
        print_acc=False):
    if not os.path.exists(experiment_folder): os.mkdir(experiment_folder)
    train_loss,train_loss0,test_loss=[],[],[]
    for epoch in range(num_epoch): 
        # model training
        model.train()
        running_loss = 0.0
        total_num=0
        for i,data in enumerate(train_loader, 0):
            inputs, labels_level1,labels_level2,labels_level3 = data
            inputs = inputs.to(device)
            labels_level3 = labels_level3.to(device)
            labels_level2 = labels_level2.to(device)
            labels_level1 = labels_level1.to(device)
            optimizer.zero_grad()
            outputs_level1,outputs_level2,outputs_level3 = model(inputs)
            loss_level3 = criterion(outputs_level3, labels_level3)
            loss_level2 = criterion(outputs_level2, labels_level2)
            loss_level1 = criterion(outputs_level1, labels_level1)
            loss=loss_level3+alpha_level2*loss_level2+alpha_level1*loss_level1
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.sum()
            total_num+=labels_level3.size(0)
        train_loss.append(-running_loss.item()/total_num)
        metrics_test=evaluate_model_multiclass(model,test_loader,criterion,num_class,device,print_acc=print_acc,model_multihead=True)
        test_loss.append(metrics_test)
        if epoch%10==0:
            pd.DataFrame(train_loss).to_csv(os.path.join(experiment_folder,'train.csv'),index=False)
        
            pd.DataFrame(test_loss).to_csv(os.path.join(experiment_folder,'test.csv'),index=False)
            torch.save(model.state_dict(), os.path.join(experiment_folder,'model'+str(epoch)+'.pt'))
    pd.DataFrame(train_loss).to_csv(os.path.join(experiment_folder,'train.csv'),index=False)
    pd.DataFrame(test_loss).to_csv(os.path.join(experiment_folder,'test.csv'),index=False)
    torch.save(model.state_dict(), os.path.join(experiment_folder,'model.pt'))
    print('Finished Training')
    return None
