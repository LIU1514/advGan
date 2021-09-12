'''
Author: your name
Date: 2021-08-25 15:19:21
LastEditTime: 2021-09-11 11:04:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \advGAN_pytorch-master\main.py
'''
import os
import pandas as pd
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
from imutils import paths
from SpeechDataGenerator_spec import SpeechDataGenerator
from utils.utils import speech_collate
from SpeechDataGenerator_spec import generate_transforms
import pretrainedmodels 
from efficientnet_pytorch import EfficientNet
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
from PIL import Image

use_cuda= False
image_nc= 3
epochs = 10
batch_size = 4
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

'''
pretrained_model = "./MNIST_target_model.pth"
targeted_model = MNIST_target_net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
'''
model_num_labels = 512

targeted_model = InceptionResnetV1(num_classes=512,device=device).eval()

### Data loaders
def generate_dataloaders(train_data,transforms): 
    
    dataset_train = SpeechDataGenerator(manifest=train_data,transforms=transforms["train_transforms"])
     

    dataloader_train = DataLoader(dataset_train, 
                                    batch_size=10, 
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=0,
                                    collate_fn=speech_collate) 
      


    print('train:', len(dataset_train))

    return dataloader_train

 

if __name__ == "__main__":

    # MNIST train dataset and dataloader declaration

    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    #dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    imagePaths = sorted(list(paths.list_images( 'images')))
    images = os.listdir('images/1')#目录里的所有文件
    transforms = generate_transforms(124)

    dataloader = generate_dataloaders(imagePaths,transforms)

    advGAN = AdvGAN_Attack(device,
                            targeted_model,
                            model_num_labels,
                            image_nc,
                            BOX_MIN,
                            BOX_MAX)

    advGAN.train(dataloader, epochs)
