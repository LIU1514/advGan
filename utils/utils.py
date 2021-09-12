'''
Author: your name
Date: 2021-07-08 10:58:10
LastEditTime: 2021-09-10 16:20:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /recognize/utils/utils.py
'''
# Third Party
import librosa
import numpy as np
import torch
import cv2
# ===============================================
#       code from Arsha for loading data.
# ===============================================



def load_data(path ):

    linear_spect = cv2.imread(path)
    
    return linear_spect

def speech_collate(batch):

    emotion=[]
    specs = []
    for sample in batch:
        specs.append(sample['spec'])
        emotion.append((sample['labels_emo']))

    return specs, emotion 
'''
fs, sr = librosa.load("data/speech1/sx217.wav") 
out=load_data("data/speech1/sx217.wav")

out=load_data("data/Train/Trainnpz/20000001_mfcc0_no_cvmn.npz")
print("run utils")
'''  
