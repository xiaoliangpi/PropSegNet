import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets
import cv2
import numpy as np
from PIL import Image
from skimage import io
import random
from typing import Sequence

from config import Config


class ImageFolderWithFilenames(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithFilenames, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        filename = os.path.basename(path)
        tuple_with_filename = (original_tuple + (filename,))
        return tuple_with_filename


class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class CropBackground:
    
    def __call__(self, img):
        img = np.array(img)
        # RGB -> BGR conversion
        img = img[...,::-1].copy()
        
        # Convert to gray and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, Config.CROP_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        # Find the max-area contour
        cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        # Crop image
        x, y, w, h = cv2.boundingRect(cnt)
        dst = img[y:y+h, x:x+w]
        dst = dst[...,::-1].copy()
        dst = dst.astype('uint8')
        return Image.fromarray(dst)

    def __repr__(self):
        return self.__class__.__name__+'()'

def get_transforms():
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(180, expand=True, fill=Config.PADDING_FILL),
        CropBackground(),
        transforms.Resize(size=Config.TARGET_SIZE, antialias=True),
        transforms.RandomVerticalFlip(Config.VERTICAL_FLIP_PROB),
        transforms.RandomHorizontalFlip(Config.HORIZONTAL_FLIP_PROB),
        transforms.Pad(padding=Config.PADDING_SIZE, fill=Config.PADDING_FILL, padding_mode='constant'),
        torchvision.transforms.ColorJitter(brightness=Config.BRIGHTNESS_FACTOR),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=Config.TARGET_SIZE, antialias=True),
        transforms.Pad(padding=Config.PADDING_SIZE, fill=Config.PADDING_FILL, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
    ])

    reshape_transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(Config.PATCH_H, Config.PATCH_W), antialias=True),
        MyRotateTransform(angles=Config.ROTATION_ANGLES),     
        torchvision.transforms.RandomVerticalFlip(Config.VERTICAL_FLIP_PROB),  
    ])

    reshape_transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(Config.PATCH_H, Config.PATCH_W), antialias=True)   
    ])

    return train_transform, test_transform, reshape_transform_train, reshape_transform_test


def get_dataloaders():
    train_transform, test_transform, _, _ = get_transforms()
    
    # Create datasets
    train_dataset = ImageFolderWithFilenames(
        root=os.path.join(Config.WORKING_DIR, 'train'),
        transform=train_transform
    )

    train_dataset_orig = ImageFolderWithFilenames(
        root=os.path.join(Config.WORKING_DIR, 'train'),
        transform=test_transform
    )
    
    train_dataset_ABD_orig = ImageFolderWithFilenames(
        root=os.path.join(Config.WORKING_DIR, 'train_ABD'),
        transform=test_transform
    )
    
    test_dataset = ImageFolderWithFilenames(
        root=os.path.join(Config.WORKING_DIR, 'test'),
        transform=test_transform
    )
    
    test_dataset_ABD = ImageFolderWithFilenames(
        root=os.path.join(Config.WORKING_DIR, 'test_ABD'),
        transform=test_transform
    )
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    trainloader_orig = torch.utils.data.DataLoader(
        train_dataset_orig, batch_size=1, shuffle=False
    )
    trainloader_ABD_orig = torch.utils.data.DataLoader(
        train_dataset_ABD_orig, batch_size=1, shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    testloader_ABD = torch.utils.data.DataLoader(
        test_dataset_ABD, batch_size=1, shuffle=False
    )
    
    return (trainloader, trainloader_orig, trainloader_ABD_orig, 
            testloader, testloader_ABD)