"""
@author:rollingball
@time:2022/11/23

一些非常实用的函数
"""
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return train_set, valid_set


# 针对不同类型的层使用不同的初始化方法
def init_weights(m):
    # 如果是全连接层
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        # if m.bias:
        #     m.bias.data.fill_(0.01)
