 
import pretrainedmodels 
from efficientnet_pytorch import EfficientNet
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loadimg =  MTCNN(
        image_size=137, margin=0, min_face_size=100,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
)
img_path = cv2.imread('./5_0.jpg')
getimg = loadimg(img_path,return_prob=True,)
model = InceptionResnetV1(num_classes=500).eval()

getID =model(getimg[0].unsqueeze(0))
print(getID.size())
'''
cv2.imshow('out',getimg[0].detach().numpy().transpose(1, 2,0 ))
cv2.waitKey(0)
print(np.argmax(getID.detach().numpy()))
'''