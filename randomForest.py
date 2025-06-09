# random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 max_features='sqrt',
                 random_state=42):
        """
        参数说明：
        - n_estimators: 森林中树的数量 (default=100)
        - max_depth: 单棵树的最大深度 (default=None，表示不限制)
        - max_features: 寻找最佳分割时考虑的特征数 (default='sqrt'表示总特征数的平方根)
        - random_state: 随机种子 (default=42)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1  # 自动使用所有CPU核心并行训练
        )

    def train(self, X_train, y_train):
        """
        训练随机森林
        输入:
        - X_train: 训练特征 (numpy数组或可转换为数组的结构)
        - y_train: 训练标签 (numpy数组)
        """
        # 自动处理输入类型转换
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        预测方法
        输入:
        - X_test: 测试特征 (numpy数组或可转换为数组的结构)
        返回:
        - 预测结果 (numpy数组)
        """
        return self.model.predict(X_test)

    def get_feature_importance(self):
        """
        获取特征重要性（仅当特征有意义时可用）
        返回:
        - 特征重要性数组 (numpy数组)
        """
        return self.model.feature_importances_