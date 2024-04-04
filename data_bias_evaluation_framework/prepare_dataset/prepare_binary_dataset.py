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
from PIL import Image, ImageDraw
from PIL import ImageFilter
import cv2
import numpy as np


class OtoDataset_Binary(Dataset):

    def __init__(self, img_df,data_dir,
                 relative_loc_col='relative_file_path',
                 subfolder_name_col='source',
                 target_col='binary_class', 
                 eclipse=False,
                 eclipse_extent=1.0,
                 transform=None):
        
        """
        Args:
            img_df (DataFrame): Dataframe including image location and its diagnosis. 
            data_dir (string): Data director
            eclipse (bool): whether to eclipse
            eclipse_extent (float): eclipse extent 
            transform (callable, optional): Transform to be applied on images.
        Returns:
            image: otoscopic image
            target: 0(normal)/1(abnormal)
        """
        self.img_df=img_df
        self.data_dir=data_dir
        self.transform = transform
        self.subfolder_name_col=subfolder_name_col
        self.relative_loc_col=relative_loc_col
        self.target_col=target_col
        self.eclipse=eclipse
        self.eclipse_extent=eclipse_extent
        
    def __len__(self):
        return self.img_df.shape[0]

    def __getitem__(self, i):
        relative_path=self.img_df[self.relative_loc_col].values[i]
        subfolder_name=self.img_df[self.subfolder_name_col].values[i]
        img_path=self.data_dir+'/'+subfolder_name+relative_path
        image = Image.open(img_path)
        target = (self.img_df[self.target_col].values)[i] 
        if self.eclipse:
            x, y =  image.size
            draw = ImageDraw.Draw(image)
            draw.ellipse( (x*(1-self.eclipse_extent)/2, y*(1-self.eclipse_extent)/2, x*(1+self.eclipse_extent)/2, y*(1+self.eclipse_extent)/2), fill=0)
        if self.transform:
            image = self.transform(image)             
        return image,target
    
