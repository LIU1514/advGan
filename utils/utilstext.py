'''
Author: your name
Date: 2021-07-08 10:58:10
LastEditTime: 2021-07-08 11:10:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /recognize/utils/utils.py
'''
# Third Party
import librosa
import numpy as np
import torch
# ===============================================
#       code from Arsha for loading data.
# ===============================================



def load_data(path, 
              spec_len=380):

 
    linear_spect = np.load( path )['x'] #(301,13)
    
    return linear_spect

def speech_collate_text(batch):
    gender = []
    emotion=[]
    specs = []
    for sample in batch:
        specs.append(sample['spec'])
        emotion.append((sample['name_id']))
        #gender.append(sample['labels_gen'])
    return specs, emotion 

