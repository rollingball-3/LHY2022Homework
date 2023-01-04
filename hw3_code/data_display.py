"""
@author:rollingball
@time:2022/11/23

1.全面观察数据类型，数据范围，数据变化情况等
进行图像化显示
"""

import os
from torchvision.io import image

# data_path = "../../hw3/food11/training/"
data_path = "../../hw3/food11/validation/"

filename_list = os.listdir(data_path)
print(filename_list)

image1 = image.read_image(os.path.join(data_path, filename_list[0]))

counter = [0] * 11
for name in filename_list:
    item_type = name.split("_")[0]
    counter[int(item_type)] += 1
print("所有标签共计")
print(counter)
