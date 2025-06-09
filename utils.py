from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Union
#评估工具
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report

from config import *


def get_dataloader(datasets_name):
    if datasets_name in ['MNIST',"FashionMNIST","KMNIST"]:
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize((0.5,), (0.5,))  # 归一化处理
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
        ])
    if datasets_name == "MNIST":
        train_dataset = datasets.MNIST(
            root='./data',  # 数据保存的路径
            train=True,  # 表示这是训练集
            transform=transform,  # 应用的转换
            download=True  # 如果数据不存在，则从互联网下载
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,  # 表示这是测试集
            transform=transform,
            download=True
        )

    elif datasets_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root='./data',  # 数据保存的路径
            train=True,  # 表示这是训练集
            transform=transform,  # 应用的转换
            download=True  # 如果数据不存在，则从互联网下载
        )

        test_dataset = datasets.FashionMNIST(
            root='./data',
            train=False,  # 表示这是测试集
            transform=transform,
            download=True
        )
    elif datasets_name == "KMNIST":
        train_dataset = datasets.KMNIST(
            root='./data',  # 数据保存的路径
            train=True,  # 表示这是训练集
            transform=transform,  # 应用的转换
            download=True  # 如果数据不存在，则从互联网下载
        )

        test_dataset = datasets.KMNIST(
            root='./data',
            train=False,  # 表示这是测试集
            transform=transform,
            download=True
        )
    elif datasets_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root='./data',  # 数据保存的路径
            train=True,  # 表示这是训练集
            transform=transform,  # 应用的转换
            download=True  # 如果数据不存在，则从互联网下载
        )

        test_dataset = datasets.CIFAR100(
            root='./data',
            train=False,  # 表示这是测试集
            transform=transform,
            download=True
        )

    elif datasets_name == "Flowers102":
        train_dataset = datasets.Flowers102(
            root='./data',  # 数据保存的路径
            split='train',  # 使用 'train'、'val' 或 'test' 来指定数据集划分
            transform=transform,  # 应用的转换
            download=True  # 如果数据不存在，则从互联网下载
        )

        test_dataset = datasets.Flowers102(
            root='./data',
            split='test',  # 使用 'train'、'val' 或 'test' 来指定数据集划分
            transform=transform,
            download=True
        )

    elif datasets_name == "Food-101":
        train_dataset = datasets.Food101(
            root='./data',  # 数据保存的路径
            split='train',  # 使用 'train'、'val' 或 'test' 来指定数据集划分
            transform=transform,  # 应用的转换
            download=True  # 如果数据不存在，则从互联网下载
        )

        test_dataset = datasets.Food101(
            root='./data',
            split='test',  # 使用 'train'、'val' 或 'test' 来指定数据集划分
            transform=transform,
            download=True
        )
    else:
        raise Exception(f"没用写这个：{datasets_name} 数据集的获取")

    train_dataloader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False)
    classes = train_dataset.classes  # 直接获取类别
    return train_dataloader, test_dataset,classes


#将tensor->numpy
def get_numpy_data(dataloader, flatten=False):
    """将PyTorch DataLoader转换为numpy数组
    支持单通道(1×H×W)和三通道(3×H×W)输入
    可选择是否展平（Flatten）图像数据

    Args:
        dataloader: PyTorch DataLoader
        flatten: 是否展平图像数据（默认False）

    Returns:
        X: numpy数组 (N, C, H, W) 或 (N, C*H*W)（如果flatten=True）
        y: numpy数组 (N,)
    """
    X_list, y_list = [], []
    for images, labels in dataloader:
        X_list.append(images.numpy())
        y_list.append(labels.numpy())

    X = np.concatenate(X_list, axis=0)  # shape: (N, C, H, W)
    y = np.concatenate(y_list, axis=0)  # shape: (N,)

    if flatten:
        # 展平操作：(N, C, H, W) -> (N, C*H*W)
        X = X.reshape(X.shape[0], -1)  # -1 自动计算 C*H*W

    return X, y

def evaluate_classifier(y_true, y_pred):
    """
    通用分类器评估函数（同时输出macro和micro指标）
    参数:
        y_true: 真实标签（形状 [n_samples]）
        y_pred: 预测标签（形状 [n_samples]）
    """
    # 基础指标（accuracy与average无关）
    accuracy = accuracy_score(y_true, y_pred)
    # 打印报告
    print("\n========== 分类器评估报告 ==========")
    print(f"准确率(Accuracy): {accuracy:.4f}")

    return accuracy,
