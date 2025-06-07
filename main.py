"""
项目：机器学习课程设计
数据集："MNIST","FashionMNIST","KMNIST","CIFAR100","Flowers102","Food-101"
"""
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# 参数
from config import *
from utils import *
# 模型


# 显示样本图像（兼容灰度和彩色）
def show_sample_image(data_loader, title):
    # 获取一个批次的数据
    images, labels = next(iter(data_loader))

    # 取批次中的第一张图像
    image = images[0]  # 现在形状是 (1, 28, 28) 或 (3, 32, 32)

    # 转换图像格式
    if len(image.shape) == 3:
        if image.shape[0] == 1:  # 灰度图像
            image = image.squeeze(0)  # 变为 (28, 28)
        elif image.shape[0] == 3:  # RGB图像
            image = image.permute(1, 2, 0)  # 变为 (32, 32, 3)

    # 显示图像
    plt.figure()
    if len(image.shape) == 2:  # 灰度图像
        plt.imshow(image.numpy(), cmap='gray')
    else:  # 彩色图像
        plt.imshow(image.numpy())

    plt.title(f"{title}\nLabel: {data_classes[labels[0].item()]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    for datasets_name in datasets_list :
        train_loader, test_loader,data_classes = get_dataloader(datasets_name)
        # 显示样本图像
        show_sample_image(train_loader, f'Sample from {datasets_name}')

