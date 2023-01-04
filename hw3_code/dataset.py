"""
@author:rollingball
@time:2022/11/23

2.对数据进行处理和筛选，制作成需要的数据集格式
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


test_tfm224 = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
    transforms.ToTensor(),
])

test_tfm128 = transforms.Compose([
    transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0)),
    transforms.ToTensor(),
])

train_tfm224 = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),

    transforms.RandomChoice([
        transforms.RandomRotation(180, resample=False, expand=False, center=None),
        transforms.RandomAffine(30),
    ]),

    transforms.RandomChoice([
        transforms.RandomGrayscale(),
        transforms.ColorJitter(0.5, 0.0, 0.0, 0.0),
        transforms.ColorJitter(0.0, 0.5, 0.0, 0.0),
        transforms.ColorJitter(0.0, 0.0, 0.5, 0.0),
        transforms.ColorJitter(0.0, 0.0, 0.0, 0.5)
    ]),

    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm128 = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),

    transforms.RandomChoice([
        transforms.RandomRotation(180, resample=False, expand=False, center=None),
        transforms.RandomAffine(30),
    ]),

    transforms.RandomChoice([
        transforms.RandomGrayscale(),
        transforms.ColorJitter(0.5, 0.0, 0.0, 0.0),
        transforms.ColorJitter(0.0, 0.5, 0.0, 0.0),
        transforms.ColorJitter(0.0, 0.0, 0.5, 0.0),
        transforms.ColorJitter(0.0, 0.0, 0.0, 0.5)
    ]),

    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm128, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files += files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        # im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label


class TestDataset(Dataset):

    def __init__(self, path, tfm1=train_tfm128, tfm2=test_tfm128, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files:
            self.files = files

        print(f"One {path} sample", self.files[0])
        self.transform1 = tfm1
        self.transform2 = tfm2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im_out = self.transform2(im).unsqueeze(0)
        # Test Time Augmentation
        for i in range(5):
            temp = self.transform1(im).unsqueeze(0)
            im_out = torch.concat((im_out, temp))

        return im_out
