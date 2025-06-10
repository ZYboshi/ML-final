import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LogisticRegressionModel:
    def __init__(self, penalty='l2', C=1.0, max_iter=1000, random_state=42):
        """
        参数说明：
        - penalty: 正则化类型 ('l1', 'l2', 'elasticnet', 'none')，默认 'l2'
        - C: 正则化强度的倒数（越小正则化越强），默认 1.0
        - max_iter: 最大迭代次数，默认 100
        - random_state: 随机种子，默认 42
        """
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='saga' if penalty == 'l1' else 'lbfgs',  # 自动选择优化器
        )

    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """返回预测类别"""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """返回预测概率（适用于概率阈值调整）"""
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test):
        """评估模型准确率"""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)