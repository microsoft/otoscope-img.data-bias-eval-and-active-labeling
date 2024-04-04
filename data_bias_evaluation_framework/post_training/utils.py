# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd
import cv2
import os
def display_image_ingrid(img_paths,ncol=None):
    if ncol is None: 
        ncol=int(len(img_paths)**0.5)
    nrow=int(np.ceil(len(img_paths)/ncol))
    im_list=[]
    for save_path in img_paths:
        image=cv2.imread(save_path)
        im=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_list.append(im)
    fig = plt.figure(figsize=(nrow,ncol))
    grid = ImageGrid(fig, 111,  
                    nrows_ncols=(nrow, ncol),  
                    axes_pad=0.02, 
                     )
    for ax, im in zip(grid,im_list):
        ax.imshow(cv2.resize(im, (112,112))  )
        ax.axis('off')
    plt.show()
    
def read_param_file(filename):
    params = {}
    with open(filename, "r") as handle:
        lines = handle.readlines()
        for line in lines:
            key, value_type_pair = line.strip().split(": ")
            value, value_type=value_type_pair.split("<")
            value=value.strip()
            if value_type == "class 'int'>":
                params[key] = int(value)
            elif value_type == "class 'float'>":
                params[key] = float(value)
            elif value_type == "class 'bool'>":
                params[key] = value == 'True'
            else:
                params[key] = value
    return params



def display_images_in_row(image_paths):
    n = len(image_paths)
    fig = plt.figure(figsize=(n,1))
    for i in range(n):
        img = mpimg.imread(image_paths[i])
        ax = fig.add_subplot(1, n, i+1)
        ax.imshow(img)
        ax.axis('off')
    plt.show()