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
from tqdm import tqdm


class HW2TrainDataset(Dataset):

    def __init__(self, frames=5) -> None:
        self.frames = frames
        label_path = "../../hw2/libriphone/libriphone/"
        train_data_path = "../../hw2/libriphone/libriphone/feat/train/"

        self.data = []
        self.label = []

        f = open(os.path.join(label_path, "train_labels.txt"))
        for line in tqdm(f, desc="处理数据集"):
            data_line = line.strip("\n").split()
            torch_data = torch.load(os.path.join(train_data_path, data_line[0] + '.pt'))
            del data_line[0]

            assert len(data_line) == torch_data.shape[0]
            length = torch_data.shape[0]

            # 边界数据怎么办
            for j in range(0, length):
                if j < frames:
                    add_torch = torch_data[0].repeat((frames - j, 1))
                    data = torch.concat((add_torch, torch_data[:j + frames + 1]), dim=0)
                elif j >= length - frames:
                    add_torch = torch_data[-1].repeat((j - length + frames + 1, 1))
                    data = torch.concat((torch_data[j - frames:], add_torch), dim=0)
                else:
                    data = torch_data[j - frames:j + frames + 1]
                self.data.append(data)
                self.label.append(int(data_line[j]))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = torch.zeros(41)
        label[self.label[index]] = 1
        return data, label


class HW2TestDataset(Dataset):

    def __init__(self, frames=5) -> None:
        self.frames = frames
        label_path = "../../hw2/libriphone/libriphone/"
        train_data_path = "../../hw2/libriphone/libriphone/feat/test/"

        self.data = []

        f = open(os.path.join(label_path, "test_split.txt"))
        for line in tqdm(f, desc="处理数据集"):
            data_line = line.strip("\n").split()
            torch_data = torch.load(os.path.join(train_data_path, data_line[0] + '.pt'))
            length = torch_data.shape[0]

            # print(torch_data.shape)
            # 边界数据怎么办
            for j in range(0, length):

                if j < frames:
                    add_torch = torch_data[0].repeat((frames - j, 1))
                    data = torch.concat((add_torch, torch_data[:j + frames + 1]), dim=0)
                elif j >= length - frames:
                    add_torch = torch_data[-1].repeat((j - length + frames + 1, 1))
                    data = torch.concat((torch_data[j - frames:], add_torch), dim=0)
                else:
                    data = torch_data[j - frames:j + frames + 1]
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        return data


class HW2RNNTrainDataset(Dataset):

    def __init__(self) -> None:

        label_path = "../../hw2/libriphone/libriphone/"
        train_data_path = "../../hw2/libriphone/libriphone/feat/train/"

        self.data = []
        self.label = []

        f = open(os.path.join(label_path, "train_labels.txt"))
        for line in tqdm(f, desc="处理数据集"):
            data_line = line.strip("\n").split()
            torch_data = torch.load(os.path.join(train_data_path, data_line[0] + '.pt'))
            del data_line[0]
            self.data.append(torch_data)
            self.label.append(data_line)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = torch.zeros((len(data), 41))
        for i, single_label in enumerate(self.label[index]):
            label[i][int(single_label)] = 1
        return data, label


class HW2RNNTestDataset(Dataset):

    def __init__(self) -> None:
        label_path = "../../hw2/libriphone/libriphone/"
        train_data_path = "../../hw2/libriphone/libriphone/feat/test/"

        self.data = []

        f = open(os.path.join(label_path, "test_split.txt"))
        for line in tqdm(f, desc="处理数据集"):
            data_line = line.strip("\n").split()
            torch_data = torch.load(os.path.join(train_data_path, data_line[0] + '.pt'))
            self.data.append(torch_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data
