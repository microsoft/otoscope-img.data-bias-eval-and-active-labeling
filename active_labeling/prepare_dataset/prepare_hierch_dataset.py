# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

class OtoDataset_Hierarchical(Dataset):

    def __init__(self, img_df,data_dir,
                 file_name_col='IMAGE_ID',
                 target_level3_col='Label_Level3_idx', 
                 target_level2_col='Label_Level2_idx',
                 target_level1_col='Label_Level1_idx', 
        self.img_df=img_df
        self.data_dir=data_dir
        self.file_name_col=file_name_col
        self.transform = transform
        self.target_level3_col=target_level3_col
        self.target_level2_col=target_level2_col
        self.target_level1_col=target_level1_col
    def __len__(self):
        return self.img_df.shape[0]

    def __getitem__(self, i):
        img_path=os.path.join(self.data_dir,self.img_df[self.file_name_col].values[i])
        image = Image.open(img_path)
        target3 = (self.img_df[self.target_level3_col].values)[i] #level3 target
        target2 = (self.img_df[self.target_level2_col].values)[i] #level2 target
        target1 = (self.img_df[self.target_level1_col].values)[i] #level1 target
        if self.transform:
            image = self.transform(image)             
        return image,target1,target2,target3
    
