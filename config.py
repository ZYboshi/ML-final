import torch

Device="cuda" if torch.cuda.is_available() else "cpu"

#pytorch参数
BatchSize=128
Epochs=20
LEARNING_RATES = [0.1, 0.01, 0.001]


#随机森林参数
n_estimators = 100
max_depth = None
max_features = 'sqrt'
random_state = 42
#全连接网络参数
Epochs_FCN = 10
#模型选择参数
Model_Type = "both"  # "pytorch", "decision_tree", "LogisticRegression" ， "Bayesian"或 "both"
#数据集选择参数
datasets_list = ["Iris","FashionMNIST","KMNIST","CIFAR10","CIFAR100","STL-10"]
datasets_name = "CIFAR100" #"FashionMNIST","KMNIST","CIFAR10","CIFAR100","STL-10"