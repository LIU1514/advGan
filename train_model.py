'''
Author: your name
Date: 2021-08-25 15:19:21
LastEditTime: 2021-09-11 19:47:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \advGAN_pytorch-master\train_target_model.py
'''
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

if __name__ == "__main__":
    use_cuda = False
    image_nc = 1
    batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    #dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    imagePaths = sorted(list(paths.list_images( 'images')))
    transforms = generate_transforms(124)

    train_dataloader = generate_dataloaders(imagePaths,transforms)

    # training the target model
    target_model = InceptionResnetV1(num_classes=500).eval()
    #target_model = MNIST_target_net().to(device)
    target_model.train()
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001)
    epochs = 1
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001)
        for i, data in enumerate(train_dataloader, 0):

            train_imgs = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in data[0]]))
            train_labels = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in data[1]])) 
 
            train_imgs, train_labels = train_imgs.to(device), train_labels.type(torch.int64).squeeze(1).to(device)

            #train_imgs, train_labels = data
            #train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)

            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()

        print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))

    # save model
    targeted_model_file_name = './IMG_target_model.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    
    '''
    target_model.eval()

    # MNIST test dataset
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('accuracy in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))
    '''