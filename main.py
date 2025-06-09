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
from network import Net
from utils import *
# 模型


"""
全局参数:train_loader \ Epochs

"""
def train_net_model(dataset_name = "CIFAR100" , input_size = 784 ,dataset_classes = 10):
    train_losses = []
    """训练PyTorch神经网络模型"""
    start_time = time.time()
    model = Net(input_size = input_size,
                num_classes=len(dataset_classes) ).to(Device)

    #决策:交叉熵
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.001)

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
        print(loss.item())
    train_end = time.time()
    train_duration = train_end - train_start


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
    print("\n******** 卷积神经网络 ********")
    evaluate_classifier(y_true, y_pred)

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
    for data_name in datasets_list :
        train_loader, test_loader,data_classes = get_dataloader(data_name)
        # 显示样本图像
        #show_sample_image(train_loader, f'Sample from {data_name}')
        image ,labels = next(iter(train_loader))
        #测试转换为Numpy

        if(data_name == "MNIST"):
            print(len(train_loader.dataset.classes))
            #输入节点数量
            input_size = image.shape[1]*image.shape[2]*image.shape[3]
            dataset_classes = train_loader.dataset.classes
            dataset_name = data_name



