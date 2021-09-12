'''
Author: your name
Date: 2021-09-07 18:04:58
LastEditTime: 2021-09-12 09:31:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \advGAN_pytorch-master\test_adversarial_examples.py
'''
 
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
from polt import showIMG
from imutils import paths
import torch
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import  MNIST_target_net
from utils.utils import speech_collate
from SpeechDataGenerator_spec import SpeechDataGenerator
from SpeechDataGenerator_spec import generate_transforms
from facenet_pytorch import MTCNN, InceptionResnetV1


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

use_cuda=False
image_nc=3
batch_size = 10

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
pretrained_model = "./MNIST_target_model.pth"
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()
target_model = InceptionResnetV1(num_classes=500).eval()

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_1.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location='cpu'))
pretrained_G.eval()

if __name__ == "__main__":

    '''
    # test adversarial examples in MNIST training dataset
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    #print(mnist_dataset[:500])
    train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct = 0
     
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        num_correct += torch.sum(pred_lab==test_label,0)

        if i == 10 :
            break

    print('MNIST training dataset:')
    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(mnist_dataset)))
    '''
    # test adversarial examples in MNIST testing dataset
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct = 0

    imagePaths = sorted(list(paths.list_images( 'images')))
    transforms = generate_transforms(124)
    train_dataloader = generate_dataloaders(imagePaths,transforms)

    for i, data in enumerate(train_dataloader, 0):
        #test_img, test_label = data
        #test_img, test_label = test_img.to(device), test_label.to(device)


        train_imgs = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in data[0]]))
        train_labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in data[1]])) 

        test_img, test_label = train_imgs.to(device), train_labels.type(torch.int64).squeeze(1).to(device)

        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        showIMG(perturbation[1])
        adv_img = perturbation + test_img
        showIMG(adv_img[1])
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        num_correct += torch.sum(pred_lab==test_label,0)

        if i == 10 :
            break

    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))

