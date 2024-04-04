# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torchvision
import torch.nn as nn
import numpy as np

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
class model_classification(nn.Module):
    def __init__(self,model_name,num_class=2):
        super(model_classification, self).__init__()  
        if model_name=="vgg16":
            model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            model.classifier[6]= Identity()
            img_size=224
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]
            num_features=4096
        elif model_name=='vit_b_16':
            model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
            num_features=model.heads.head.in_features
            model.heads.head=Identity()
            img_size=224
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]

        elif model_name=='vit_b_16_384':
            model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            num_features=model.heads.head.in_features
            model.heads.head=Identity()
            img_size=384
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]
        elif model_name=='vit_l_16':
            model = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.DEFAULT)
            num_features=model.heads.head.in_features
            model.heads.head=Identity()
            img_size=224
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]
        elif model_name=='vit_h_14':
            model = torchvision.models.vit_h_14(weights=torchvision.models.ViT_H_14_Weights.DEFAULT)
            num_features=model.heads.head.in_features
            model.heads.head=Identity()
            img_size=518
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]
        elif model_name=='swin_v2_b':
            model = torchvision.models.swin_v2_b(weights=torchvision.models.Swin_V2_B_Weights.DEFAULT)
            num_features=model.head.in_features
            model.heads.head=Identity()
            img_size=256
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]
        elif model_name=='swin_v2_s':
            model = torchvision.models.swin_v2_s(weights=torchvision.models.Swin_V2_S_Weights.DEFAULT)
            num_features=model.head.in_features
            model.heads.head=Identity()
            img_size=256
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            num_features=2048
            model.fc=Identity()
            img_size=224
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]
        elif model_name == 'densenet161':
            model=torchvision.models.densenet161(weights=torchvision.models.DenseNet161_Weights.DEFAULT)
            num_features=2208
            model.classifier=Identity()
            img_size=224
            model_mean = [0.485, 0.456, 0.406] 
            model_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError("Invalid model name!")
        last_layer=nn.Linear(num_features, num_class)
        self.model=model
        self.size=img_size
        self.mean=model_mean
        self.std=model_std
        self.last_layer=last_layer
    def __size__(self):
        return self.size
    def __mean__(self):
        return self.mean
    def __std__(self):
        return self.std
    def forward(self, img):
        out = self.model(img)
        out=self.last_layer(out)
        return out