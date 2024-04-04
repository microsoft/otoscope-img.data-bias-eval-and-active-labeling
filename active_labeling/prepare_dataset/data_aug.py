# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import torch
from  torchvision.transforms.functional import InterpolationMode
import numpy as np

class GaussnoiseTransform(object):
    def __init__(self, mean=0, sigma=0.1, random_seed=None):
        self.mean=mean
        self.sigma=sigma
        self.random_seed=random_seed
    def __call__(self, img):
        img= np.asarray(img)
        if self.random_seed:
            np.random.seed(self.random_seed)
        gauss = np.random.normal(self.mean,self.sigma,img.shape)
        img_noised=img+gauss
        img_noised=img_noised.astype(np.uint8)
        return Image.fromarray(img_noised)


def derive_transform(size,mean,std,scale=.8,
                     add_gauss_noise=False,
                     elastic_tf=True,
                     color_hue=0.05,elastic_tf_alpha=50.0):
    augs={
        "size":0,
        "mean":0,
        "std":0,
       "color_contrast": 0.1, 
       "color_saturation": 0.1, 
       "color_brightness": 0.1, 
       "rotation": 90, 
       "shear": 20}
    augs['color_hue'] = color_hue
    augs['size'] = size
    augs['mean'] = mean
    augs['std'] = std
    tf_list = []
    interpolation=InterpolationMode.BICUBIC
    tf_list.append(transforms.RandomResizedCrop(augs['size'], scale=(scale, 1),interpolation=interpolation))
    tf_list.append(transforms.RandomHorizontalFlip())
    tf_list.append(transforms.RandomVerticalFlip())
    tf_list.append(transforms.ColorJitter(
        brightness=augs['color_brightness'],
        contrast=augs['color_contrast'],
        saturation=augs['color_saturation'],
        hue=augs['color_hue']))
    if elastic_tf: tf_list.append(transforms.ElasticTransform(alpha=elastic_tf_alpha))
    if add_gauss_noise: tf_list.append(GaussnoiseTransform())
    tf_list.append(transforms.ToTensor())
    tf_augment = transforms.Compose(tf_list)
    train_tf=transforms.Compose([tf_augment,transforms.Normalize(augs['mean'], augs['std'])])
    
    orig_tf=transforms.Compose([transforms.Resize((augs['size'],augs['size']),interpolation=interpolation),transforms.ToTensor(),transforms.Normalize(augs['mean'], augs['std'])])
    return train_tf,orig_tf