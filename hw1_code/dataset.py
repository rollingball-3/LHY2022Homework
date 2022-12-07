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


# high_corr_index = list(range(116))

# 0.8
high_corr_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 52, 53, 54, 55, 56, 68, 69, 70, 71, 72, 84, 85,
                   86, 87, 88, 100, 101, 102, 103, 104]

# 0.6
# high_corr_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
#                    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 49, 52, 53, 54, 55, 56, 65, 68, 69, 70, 71, 72,
#                    81, 84, 85, 86, 87, 88, 97, 100, 101, 102, 103, 104, 107, 113]

# 0.4
# high_corr_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
#                    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 48, 49, 50, 52, 53, 54, 55, 56, 59, 64, 65,
#                    66, 68, 69, 70, 71, 72, 75, 80, 81, 82, 84, 85, 86, 87, 88, 91, 96, 97, 98, 100, 101, 102, 103, 104,
#                    107, 112, 113, 114]

# 37个州
def process_data(train_data_path="../../hw1/covid.train.csv", test_data_path="../../hw1/covid.test.csv"):
    train_x, train_y = process_train_data(train_data_path)
    test_x = process_test_data(test_data_path)

    # # 为什么用normalization效果变差了呢？
    # train_len = len(train_x)
    # concat_x = np.concatenate((train_x, test_x), axis=0)
    # concat_x[:, 37:] = (concat_x[:, 37:] - np.mean(concat_x[:, 37:], axis=0, keepdims=True)) / \
    #                    np.std(concat_x[:, 37:], axis=0, keepdims=True)

    # train_x = concat_x[:train_len]
    # test_x = concat_x[train_len:]

    return train_x, train_y, test_x


def process_train_data(train_data_path="../../hw1/covid.train.csv"):
    train_data = pd.read_csv(train_data_path)
    train_data.drop(['id'], axis=1, inplace=True)

    train_data = np.array(train_data)
    y = train_data[:, -1]
    x = train_data[:, high_corr_index]

    return x, y


def process_test_data(test_data_path="../../hw1/covid.test.csv"):
    test_data = pd.read_csv(test_data_path)
    test_data.drop(['id'], axis=1, inplace=True)

    test_data = np.array(test_data)
    x = test_data[:, high_corr_index]
    return x


class HW1Dataset(Dataset):

    def __init__(self, x, y=None) -> None:
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]
