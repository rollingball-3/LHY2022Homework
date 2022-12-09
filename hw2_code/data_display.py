"""
@author:rollingball
@time:2022/11/23

1.全面观察数据类型，数据范围，数据变化情况等
进行图像化显示
"""

import pandas as pd
import numpy as np
import torch
import os

data_path = "/home/scu-108/qfz/kaggle/hw_LHY_local/hw2/libriphone/libriphone/feat/train"

print("语音数据:")
file_list = os.listdir(data_path)
print(file_list)
data_temp = torch.load(os.path.join(data_path, file_list[0]))
print(data_temp.shape)
print(data_temp)

print("label:")
path = "/home/scu-108/qfz/kaggle/hw_LHY_local/hw2/libriphone/libriphone/"
train_split = "/home/scu-108/qfz/kaggle/hw_LHY_local/hw2/libriphone/libriphone/train_split.txt"
train_labels = "/home/scu-108/qfz/kaggle/hw_LHY_local/hw2/libriphone/libriphone/train_labels.txt"

f = open(train_labels)
line_data = f.readline().split()

print(line_data)
print(len(line_data))
sound_data = torch.load(os.path.join(data_path, line_data[0] + '.pt'))
print(sound_data.shape)
