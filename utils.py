from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Union
#评估工具
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

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
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化处理
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
    elif datasets_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root='./data',  # 数据保存的路径
            train=True,  # 表示这是训练集
            transform=transform,  # 应用的转换
            download=True  # 如果数据不存在，则从互联网下载
        )

        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,  # 表示这是测试集
            transform=transform,
            download=True
        )
    elif datasets_name == "Iris":
        iris = load_iris()
        X, y = iris.data, iris.target  # X: 特征 (150x4), y: 标签 (150,)
        print("总样本数:", X.shape[0])  # 输出: 150
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,  # 验证集比例
            random_state=42,  # 随机种子（确保可复现）
            stratify=y  # 保持类别比例一致
        )
        print("训练集:", X_train.shape[0], "验证集:", X_val.shape[0])
        # 转为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        # 封装为Dataset和DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        classes = iris.target_names.tolist()
        train_dataset.classes = classes
        # 可视化（选择两个特征）
        plt.figure(figsize=(10, 6))
        for i, class_name in enumerate(classes):
            plt.scatter(X[y == i, 0], X[y == i, 2], label=class_name, alpha=0.7)
        plt.xlabel(iris.feature_names[0] + " (cm)")
        plt.ylabel(iris.feature_names[2] + " (cm)")
        plt.title("Iris Dataset Visualization (Sepal vs Petal Length)")
        plt.legend()
        plt.grid(True)
        plt.show()
    elif datasets_name == "STL-10":
        train_dataset = datasets.STL10(
            root='./data', split='train', transform=transform, download=True
        )
        test_dataset = datasets.STL10(
            root='./data',
            split='test', transform=transform,
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
