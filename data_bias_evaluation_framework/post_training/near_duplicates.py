# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch.nn as nn
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import utils
def merge_lists(aa,agressive=True):
    out = []
    while len(aa) > 0:
        first, *rest = aa
        first = set(first)
        lf = -1
        while len(first) > lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if agressive:
                    if len(first.intersection(set(r))) > 0:
                        first |= set(r)
                    else:
                        rest2.append(r)
                else: 
                    if first.issubset(set(r)) or set(r).issubset(first):
                        first |= set(r)
                    else:
                        rest2.append(r)
            rest = rest2
        out.append(list(first))
        aa = rest
    return out



def return_nearduplicate_sets(samples,
                              alpha=0.8,
                              merge_agressive=True  # if True, merge the near duplicate sets if there is at least one common element. Otherwise, merge the near duplicate sets if one is a subset of the other.
                              ):
    neigh = NearestNeighbors(n_neighbors=10, metric="cosine")
    neigh.fit(samples)
    distances, indices = neigh.kneighbors(samples,n_neighbors=2)
    median_distances = np.median(distances[:,1:])
    th=median_distances*alpha
    radius_neigh=neigh.radius_neighbors(samples, radius=th, return_distance=False)
    near_duplicate_collection=[]
    for ii in range(len(radius_neigh)):
        if len(radius_neigh[ii])>1:
            near_duplicate_cases = set(radius_neigh[ii])
            near_duplicate_collection+=[(near_duplicate_cases)]
    near_duplicate_set=[list(x) for x in set(tuple(x) for x in near_duplicate_collection)]

    return merge_lists(near_duplicate_set,agressive=merge_agressive)

def return_embeddings(model0,data_loader,device):
    embedding_list=[]
    model=model0.model
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            out=torch.flatten(outputs, 1).cpu().numpy().tolist()
            embedding_list+=out
    return embedding_list


def return_near_duplicate_set_list(data_origin,img_df,alpha=0.4,merge_agressive=True):
   near_duplicate_set_list=[]
   sub_df=img_df[img_df['source']==data_origin]
   index_dict=dict(zip(range(sub_df.shape[0]),sub_df.index))
   for test_fold_id in range(5):
      embedding_df=pd.read_csv(f'../metadata/embedding{test_fold_id}.csv')
      embedding_sub_df=embedding_df[embedding_df.index.isin(index_dict.values())]
      embedding_list=embedding_sub_df.values.tolist()
      near_duplicate_set=return_nearduplicate_sets(embedding_list,
                                 alpha=alpha,
                                 merge_agressive=merge_agressive)
      near_duplicate_set_list+=near_duplicate_set
   merged_duplicate_sets=merge_lists(near_duplicate_set_list,agressive=merge_agressive)
   merged_duplicate_sets=[ [index_dict[x]  for x in x_list] for x_list in merged_duplicate_sets]
   df=near_duplicate_set_stat(merged_duplicate_sets,data_origin)
   return merged_duplicate_sets,df
def near_duplicate_set_stat(merged_duplicate_sets,data_origin):
    set_size_list=[len(x) for x in merged_duplicate_sets]
    set_str_list=[','.join(map(str,x)) for x in merged_duplicate_sets]
    data_origin_list=[data_origin]*len(merged_duplicate_sets)
    near_duplicate_set_df=pd.DataFrame({'set':set_str_list,'data_origin':data_origin_list,'set_size':set_size_list})
    return near_duplicate_set_df

def show_top_near_duplicate_set(near_duplicate_set_df,img_df,DATA_DIR,
top_n=20,print_class_distribution=False):
    top_n=min(top_n,near_duplicate_set_df.shape[0])
    top_index_list=near_duplicate_set_df.nlargest(top_n, 'set_size').index
    for ii in top_index_list:
        neardup_set=near_duplicate_set_df['set'].values[ii]
        neardup_set=[int(x) for x in neardup_set.split(',')]
        relative_path=img_df['relative_file_path'].values[neardup_set]
        subfolder_name=img_df['source'].values[neardup_set]
        img_paths=DATA_DIR+'/'+subfolder_name+relative_path
        if len(img_paths)<=20:
            utils.display_images_in_row(img_paths)
        elif len(img_paths)<=100:
            utils.display_image_ingrid(img_paths)
        else: 
            print('100 images are randomly selected for display.')
            sample_img_paths=np.random.choice(img_paths, size=100, replace=False)
            utils.display_image_ingrid(sample_img_paths)
        if print_class_distribution:
            print('Label distribution: ',img_df.iloc[neardup_set]['class'].value_counts())