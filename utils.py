from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Union
from typing import Dict, List
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


def plot_algorithm_comparison(results: Dict[str, Dict[str, float]], save_path: str = "algorithm_comparison.png"):
    """
    绘制三个算法在不同数据集上的准确率对比图

    参数:
        results: 字典，包含每个数据集下三个算法的准确率
            格式: {
                "数据集1": {
                    "神经网络": 0.95,
                    "随机森林": 0.90,
                    "逻辑回归": 0.85
                },
                "数据集2": {...},
                ...
            }
        save_path: 图片保存路径
    """
    # 准备数据
    datasets = list(results.keys())
    algorithms = ["神经网络", "随机森林", "逻辑回归"]

    # 为每个算法准备准确率数据
    accuracy_data = {
        "神经网络": [results[dataset]["神经网络"] for dataset in datasets],
        "随机森林": [results[dataset]["随机森林"] for dataset in datasets],
        "逻辑回归": [results[dataset]["逻辑回归"] for dataset in datasets]
    }

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 设置柱状图宽度和位置
    bar_width = 0.25
    index = np.arange(len(datasets))

    # 绘制每个算法的柱状图
    for i, (algo, accuracies) in enumerate(accuracy_data.items()):
        plt.bar(index + i * bar_width, accuracies, bar_width, label=algo)

    # 添加标签和标题
    plt.xlabel('数据集')
    plt.ylabel('准确率')
    plt.title('不同算法在各数据集上的准确率比较')
    plt.xticks(index + bar_width, datasets, rotation=45)
    plt.legend()

    # 添加数值标签
    for i, dataset in enumerate(datasets):
        for j, algo in enumerate(algorithms):
            height = results[dataset][algo]
            plt.text(index[i] + j * bar_width, height + 0.01, f"{height:.3f}",
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def display_algorithm_comparison_per_dataset(results: Dict[str, Dict[str, float]]):
    """
    为每个数据集单独显示三个算法的准确率对比折线图

    参数:
        results: 字典，包含每个数据集下三个算法的准确率
            格式: {
                "数据集1": {
                    "神经网络": 0.95,
                    "随机森林": 0.90,
                    "逻辑回归": 0.85
                },
                "数据集2": {...},
                ...
            }
    """
    # 为每个数据集单独显示图表
    for dataset_name, accuracies in results.items():
        plt.figure(figsize=(10, 6))

        # 准备数据
        algorithms = list(accuracies.keys())
        accuracy_values = list(accuracies.values())

        # 绘制折线图
        plt.plot(algorithms, accuracy_values, marker='o', linestyle='-',
                 linewidth=2, markersize=8, color='royalblue')

        # 添加数值标签
        for algo, acc in accuracies.items():
            plt.text(algo, acc + 0.01, f"{acc:.3f}",
                     ha='center', va='bottom', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 设置图表属性
        plt.title(f"{dataset_name} - 算法准确率比较", fontsize=14, pad=20)
        plt.xlabel('算法', fontsize=12)
        plt.ylabel('准确率', fontsize=12)
        plt.ylim(0, 1.1)  # 准确率范围固定在0-1.1
        plt.grid(True, linestyle='--', alpha=0.6)

        # 调整布局
        plt.tight_layout()

        # 显示图表
        plt.show()

#loss曲线绘制
def plot_loss_curve(epochs_list: List[int],
                    train_losses: List[float],
                    dataset_name: str = "",
                    ) -> None:

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, train_losses, 'b-o',
             linewidth=2,
             markersize=8,
             label="Training Loss")

    plt.title(f'Training Loss ({dataset_name})', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.show()

if __name__ == "__main__":
    # 这里应该是您实际运行算法后收集的结果
    # 示例数据 - 请替换为您实际的运行结果
    example_results = {
        "Iris": {
            "神经网络": 0.3333,
            "随机森林": 0.9667,
            "逻辑回归": 0.9667
        },
        "FashionMNIST": {
            "神经网络": 0.7899,
            "随机森林": 0.8761,
            "逻辑回归": 0.8392
        },
        "KMNIST": {
            "神经网络": 0.5254,
            "随机森林": 0.8584,
            "逻辑回归": 0.6902
        },
        "CIFAR100": {
            "神经网络": 0.0557,
            "随机森林": 0.2244,
            "逻辑回归": 0.1191
        },
        "CIFAR10": {
            "神经网络": 0.2861,
            "随机森林": 0.4680,
            "逻辑回归": 0.3663
        },
        "STL-10": {
            "神经网络": 0.1829,
            "随机森林": 0.4078,
            "逻辑回归": 0.2925
        }
    }
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plot_algorithm_comparison(example_results)
    display_algorithm_comparison_per_dataset(example_results)