"""
@author:rollingball
@time:2022/11/23

1.全面观察数据类型，数据范围，数据变化情况等
进行图像化显示
"""

import pandas as pd
import numpy as np

# 探究数据结构

train_file_path = "../../hw1/covid.train.csv"
test_file_path = "../../hw1/covid.test.csv"

train_data = pd.read_csv(train_file_path)
train_data.info()
test_data = pd.read_csv(test_file_path)
test_data.info()

for i, key in enumerate(train_data.keys()):
    print(i, key)

for key in train_data.keys():
    if key not in test_data.keys():
        print(key)

# print(1 + 37 + 5 * (4 + 8 + 3 + 1))
print("########################################################")

# 探究数据相关性
no_states_data = train_data.drop(train_data.keys()[0:38], axis=1)

corr = no_states_data.corr()

# print(corr['tested_positive.4'])

feature = abs(corr['tested_positive.4']) > 0.8
feature_list = [i for i in range(37)]
for i, k in enumerate(feature):
    if k:
        feature_list.append(i + 37)

print(feature_list)

# print(corr.loc[abs(corr['tested_positive.4']) > 0.4, "tested_positive.4"])
# print(corr.sort_values(by='tested_positive.4')['tested_positive.4'])
