"""
@author:rollingball
@time:2022/11/23

3.定义总的网络结构
"""
import torch
import torch.nn as nn


class HW2_net(nn.Module):

    def __init__(self, input_dim, output_dim, use_BatchNorm=True, dropout=0.25):
        super(HW2_net, self).__init__()

        nn_list = []
        for i in range(6):
            if i == 0:
                if use_BatchNorm:
                    nn_list.append(nn.Linear(input_dim, 1024, bias=False))
                    nn_list.append(nn.BatchNorm1d(1024))
                else:
                    nn_list.append(nn.Linear(input_dim, 1024))
            else:
                if use_BatchNorm:
                    nn_list.append(nn.Linear(1024, 1024, bias=False))
                    nn_list.append(nn.BatchNorm1d(1024))
                else:
                    nn_list.append(nn.Linear(1024, 1024))

            nn_list.append(nn.ReLU())

            if dropout > 0:
                nn_list.append(nn.Dropout(dropout))

        nn_list.append(nn.Linear(1024, output_dim))

        self.net = nn.Sequential(*nn_list)

    def forward(self, d):
        batch_size = d.shape[0]
        d = d.view(batch_size, -1)
        out = self.net(d)
        return out


class HW2RNN_net(nn.Module):

    def __init__(self, input_dim=39, output_dim=100, layers_number=2, dropout=0.25):
        super(HW2RNN_net, self).__init__()

        self.lstm = nn.LSTM(input_dim, output_dim, layers_number, dropout=dropout, batch_first=True)

        self.linear = nn.Linear(output_dim, 41)

    def forward(self, d):
        output, (hn, cn) = self.lstm(d)
        output = output.squeeze()
        out = self.linear(output)
        return out
