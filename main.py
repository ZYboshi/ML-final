"""
项目：机器学习课程设计
数据集："MNIST","FashionMNIST","KMNIST","CIFAR100","Flowers102","Food-101"
"""
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import numpy as np
# 参数
from config import *
from logisticregress import LogisticRegressionModel
from utils import *
# 模型
from network import Net
from randomForest import RandomForest
#预测
from sklearn.metrics import accuracy_score
"""
全局参数:train_loader \ Epochs

"""
def train_net_model(dataset_name = "CIFAR100" , train_loader: Optional[DataLoader] = None ,test_loader: Optional[DataLoader] = None):
    #数据处理:+输入节点+类型数量
    print(f'>>>>>>全连接网络***{dataset_name}<<<<<<')
    image, labels = next(iter(train_loader))
    if dataset_name == "Iris":
        input_size = image.shape[1]
    else:
        input_size = image.shape[1] * image.shape[2] * image.shape[3]
    dataset_classes = train_loader.dataset.classes

    train_losses = []
    epochs_list = []  # 记录epoch编号

    """训练PyTorch神经网络模型"""
    start_time = time.time()
    model = Net(input_size = input_size,
                num_classes=len(dataset_classes) ).to(Device)

    #决策:交叉熵
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.01)

    #训练
    train_start = time.time()
    for epoch in range(Epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(Device), target.to(Device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"\nEpoch[{epoch}/{Epochs}]\tLoss: {loss.item():.6f}\n")
        train_losses.append(loss.item())
        epochs_list.append(epoch)
        print(loss.item())
    train_end = time.time()
    train_duration = train_end - train_start

    plot_loss_curve(epochs_list, train_losses,dataset_name)

    #测试
    test_start = time.time()
    model.eval()
    y_pred, y_true = [], []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(Device), target.to(Device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        y_pred.extend(pred.cpu().numpy().flatten())
        y_true.extend(target.cpu().numpy().flatten())
    test_end = time.time()
    test_duration = test_end - test_start

    total_time = time.time() - start_time
    print("\n******** 神经网络 ********")
    evaluate_classifier(y_true, y_pred)

#随机森林
def train_randomForest(dataset_name = "MNIST",
                       train_loader: Optional[DataLoader] = None,
                        test_loader: Optional[DataLoader] = None,):
    #数据处理:获得numpy数据+输入节点+类型数量
    print(f'>>>>>>随机森林***{dataset_name}<<<<<<')
    image, labels = next(iter(train_loader))
    if dataset_name == "Iris":
        input_size = image.shape[1]
    else:
        input_size = image.shape[1] * image.shape[2] * image.shape[3]
    dataset_classes = train_loader.dataset.classes
    X_train, y_train = get_numpy_data(train_loader, flatten=True)
    X_test, y_test = get_numpy_data(test_loader, flatten=True)
    #模型建立
    rf = RandomForest(n_estimators = n_estimators,
                        max_depth = max_depth,
                        max_features = max_features,
                        random_state = random_state,)
    rf.train(X_train, y_train)
    # 预测测试集
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

#逻辑回归
def train_logisticRegression(dataset_name = "MNIST",
                             train_loader: Optional[DataLoader] = None,
                             test_loader: Optional[DataLoader] = None,):
    #数据处理:获得numpy数据+输入节点+类型数量
    print(f'>>>>>>逻辑回归***{dataset_name}<<<<<<')
    image, labels = next(iter(train_loader))
    if dataset_name == "Iris":
        input_size = image.shape[1]
    else:
        input_size = image.shape[1] * image.shape[2] * image.shape[3]
    dataset_classes = train_loader.dataset.classes
    X_train, y_train = get_numpy_data(train_loader, flatten=True)
    X_test, y_test = get_numpy_data(test_loader, flatten=True)


    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")


#显示样本图像（兼容灰度和彩色）
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
#测试是否正确
def try_data(dataset_name = "MNIST",
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,):
    print(f'{dataset_name}')
    image, labels = next(iter(train_loader))
    # 输入节点数量
    if dataset_name == "Iris":
        input_size = image.shape[1]
        print(input_size)
    else:
        input_size = image.shape[1] * image.shape[2] * image.shape[3]
    dataset_classes = train_loader.dataset.classes
    dataset_name = data_name
    print(f'输入节点：{input_size} \n 种类数:{dataset_classes}')
    #数据转换为numpy
    X_train , y_train = get_numpy_data(train_loader,flatten=True)
    X_test, y_test = get_numpy_data(test_loader,flatten=True)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
if __name__ == "__main__":
    for data_name in datasets_list :
        train_loader, test_loader,data_classes = get_dataloader(data_name)
        train_randomForest(dataset_name = data_name,train_loader = train_loader,test_loader = test_loader)
        train_logisticRegression(dataset_name = data_name,train_loader = train_loader,test_loader = test_loader)
        train_net_model(data_name,train_loader,test_loader)
        # # 显示样本图像
        # if data_name != "Iris":
        #     show_sample_image(train_loader, f'Sample from {data_name}')
        # image ,labels = next(iter(train_loader))
        # try_data(data_name,train_loader,test_loader)
        # #测试转换为Numpy
        #
        # if(data_name == "MNIST"):
        #     print(len(train_loader.dataset.classes))
        #     #输入节点数量
        #     input_size = image.shape[1]*image.shape[2]*image.shape[3]
        #     dataset_classes = train_loader.dataset.classes
        #     dataset_name = data_name



