# 机器学习实验课程设计
> 西南交通大学  2022116481  张彦博
>
>MNIST手写数字识别数据集在不同模型上的比较

本项目实现了多种机器学习模型在MNIST手写数字识别任务上的训练和预测，包括：

* `卷积神经网络 (CNN)`
* `全连接神经网络 (FCN)`
* `决策树`
* `逻辑回归`
* `贝叶斯分类器`

## 模型训练准备
1. 数据准备
项目使用MNIST数据集，包含
* 60,000张训练图像
* 10,000张测试图像
数据加载和预处理通过`utils.py`中的`get_dataloader`函数完成。

2. 训练配置
训练参数在`config.py`中配置，其中模型选择参数`Model_Type`选择为both的时候即为所有模型都执行：  
```python  
# 公共参数
# 决策树参数
# 全连接网络参数

# 模型选择
Model_Type = "both"  # 可选："pytorch", "decision_tree", "LogisticRegression", "Bayesian", "FCN" 或 "both"
```

## 模型训练
>模型训练以及预测都写在同一个函数中，通过调用`main.py`中train_xxx函数即可进行模型训练以及预测

1. 运行`main.py`进行训练+预测

| 模型 | 实现文件 | 训练函数 | 备注 |
|------|----------|----------|------|
| ​**卷积神经网络(CNN)​**​ | `network.py` | `train_pytorch_model(lr=0.01, optimizer_type='adam')` | 支持多种优化器 |
| ​**全连接神经网络(FCN)​**​ | `FCNnet.py` | `train_fully_connected()` | 单层全连接网络 |
| ​**决策树**​ | `decisiontree.py` | `train_decision_tree()` | 基于scikit-learn实现 |
| ​**逻辑回归**​ | `logisticregress.py` | `train_LogisticRegression()` | 多分类逻辑回归 |
| ​**贝叶斯分类器**​ | `bayesian.py` | `train_BayesianClassifier()` | 高斯朴素贝叶斯 |

2. 卷积网络模型权重参数
在`main.py`中 卷积网络训练过程中保存了卷积网络模型权重参数，用于断点续训等操作,保存在文件夹:`checkpoints`  
代码为:
```python
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}")
            torch.save(model.state_dict(), f"./checkpoints/fc_model_{epoch}.pth")
```  
## 项目地址
GitHub仓库：https://github.com/ZYboshi/img-classify.git