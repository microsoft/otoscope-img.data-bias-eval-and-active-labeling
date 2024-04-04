# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pandas as pd

def write_file(filename, **kwargs):
    with open(filename, "w") as handle:
        for key, value in kwargs.items():
            handle.write("{}: {} {} \n" .format(key, value, type(value)))
def generate_class_idx_dict(df_col):
    all_class_list=df_col.unique().tolist()
    all_class_list.sort()
    class_index_list=list(range(0,len(all_class_list)))
    class_to_idx=dict(zip(all_class_list, class_index_list))
    return class_to_idx   
def generate_level2_class(level3_class_value):
    if level3_class_value=='INCONCLUSIVE': return 'INCONCLUSIVE'
    if level3_class_value=='FB_BODY' or level3_class_value=='GROMMET': return 'OBJECT_IN_EAR'
    if level3_class_value=='WET' or level3_class_value=='DRY': return 'PERFORATION'
    if level3_class_value=='INTACT_ABNORMAL' or level3_class_value=='INTACT_NORMAL': return 'INTACT'
    print('Class not found. Return None!',level3_class_value) 
    return None

def generate_level1_class(level2_class_value):
    if level2_class_value=='INCONCLUSIVE': return 'INCONCLUSIVE'
    if level2_class_value=='OBJECT_IN_EAR' or level2_class_value=='PERFORATION' or level2_class_value=='INTACT': return 'CONCLUSIVE'
    print('Class not found. Return None!') 
    return None