'''
Author: your name
Date: 2021-09-03 16:39:41
LastEditTime: 2021-09-07 21:21:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \advGAN_pytorch-master\polt.py
'''
import cv2

def showIMG(data_img):
    data_img = data_img.detach().numpy()
    data_img = cv2.resize(data_img.transpose(2, 1,0 ),(200,200))
    #cv2.imshow('out',data_img.transpose(2, 1,0 ))
    cv2.imshow('out',data_img )
    cv2.waitKey(0)

 