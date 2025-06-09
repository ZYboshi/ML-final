import torch

Device="cuda" if torch.cuda.is_available() else "cpu"

#pytorch参数
BatchSize=1
Epochs=20
LEARNING_RATES = [0.1, 0.01, 0.001]


#决策树参数
max_depth = 10
#全连接网络参数
Epochs_FCN = 10
#模型选择参数
Model_Type = "both"  # "pytorch", "decision_tree", "LogisticRegression" ， "Bayesian"或 "both"
#数据集选择参数
datasets_list = ["MNIST","FashionMNIST","KMNIST","CIFAR100","Flowers102","Food-101"]
datasets_name = "CIFAR100" #'MNIST',"FashionMNIST","KMNIST" "CIFAR100"  "Flowers102"  "Food-101"