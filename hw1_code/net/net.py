"""
@author:rollingball
@time:2022/11/23

3.定义总的网络结构
"""

import torch.nn as nn


class HW1_net(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_number=None):
        super(HW1_net, self).__init__()

        if dropout_number:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_number),
                nn.Linear(hidden_dim, output_dim),
            )

        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, d):
        out = self.net(d)
        out = out.squeeze(1)
        return out
