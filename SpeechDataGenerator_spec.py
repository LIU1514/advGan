'''
Author: your name
Date: 2021-06-15 10:23:30
LastEditTime: 2021-09-11 10:35:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /speech-person-recognition/SpeechDataGenerator_spec.py
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar  4 21:20:52 2020

@author: krishna
"""
import cv2
import numpy as np
import torch
from PIL import Image
from utils import utils
from torch.utils.data import Dataset   
#from torchvision import transforms as T
from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    Resize,
    ShiftScaleRotate,
    VerticalFlip,
    RandomGridShuffle,
    InvertImg,
    RandomBrightnessContrast,
    CoarseDropout,
    HueSaturationValue
)

'''
train_transform = Compose(
                    [

                        VerticalFlip(p=0.5),
                        HorizontalFlip(p=0.5),
                        ShiftScaleRotate(
                            shift_limit=0.2,
                            scale_limit=0.2,
                            rotate_limit=20,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_REFLECT_101,
                            p=1, )
                    ]
    )
'''

def generate_transforms(image_size ):

    train_transform = Compose(
        [

            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Resize(height=100, width=200),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            #RandomGridShuffle(grid=(2, 2), p=1),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
            InvertImg(always_apply=False, p=0.6),
            OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
            OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
            #CoarseDropout(p=0.5),
            ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=45,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1, ),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            #ToTensorV2(p=1.0),
                
              
        ]
    )

    val_transform = Compose(
        [
            #ToTensorV2(p=1.0),
            Resize(height=100, width=200),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )

    return {"train_transforms": train_transform, "val_transforms": val_transform}


class SpeechDataGenerator(Dataset ):
    """Speech dataset."""

    def __init__(self, manifest, mode='train', transforms=None):
        
        """
        Read the textfile and get the paths
        """
        self.audio_links = manifest 
        self.labels_emotion = manifest
        #print(len(self.audio_links))
        #self.labels_gender = [int(line.rstrip('\n').split(' ')[2]) for line in open(manifest)]
        '''
        self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    #T.RandomVerticalFlip( ),
                ])     

        '''
 
        self.transforms =  transforms

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link =self.audio_links[idx]

        get_id = self.labels_emotion[idx]
        emo_id = int(get_id.split('\\')[1] )  
        
        #gen_id = self.labels_gender[idx]
        specgram = utils.load_data(audio_link)
        specgram = cv2.medianBlur(specgram,3)
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]], np.float32)
        specgram  = cv2.filter2D(specgram, -1, kernel=kernel)

        specgram = cv2.resize(specgram, (114,114 ))

        
        #specgram = np.squeeze(specgram)
        #specgram = Image.fromarray(specgram)

         

        # Do data augmentation .transpose(2, 0, 1)
        if self.transforms is not None:
            specgram = self.transforms(image=specgram)["image"].transpose(2, 0, 1)
            
        #specgram = np.transpose(specgram,(0,1))
        #specgram = np.expand_dims(specgram, 0)
        
        #原来specgram(384,300)
        #specgram = specgram.reshape(1,specgram.shape[0], specgram.shape[1]
        #         ).astype("float32")   
         
        #specgram = specgram.transpose(0, 1,2)
        #specgram = cv2.resize(specgram, (300,200 ))

        
        sample = {'spec': torch.from_numpy(np.ascontiguousarray(specgram)), 
                'labels_emo': torch.from_numpy(np.ascontiguousarray(emo_id))}        
        return sample

    print("over")

# np.ascontiguousarray() 数据连接