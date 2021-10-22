"""
Navie bayes algorithms.
"""
import numpy as np
from collections import Counter
class GaussianNB:
    #todo(felix): 补全注释
    """
    GaussianNB Class.
    Parameters
    ----------
    var_smoothing : float, default=1e-09
        Gaussian smoothing parameter.
    Attributes
    ----------
    class_priors : list of shape (n_classes)
        Frequency of each category in the training set.
    """
    def __init__(self, var_smoothing=1e-09):
        self.var_smoothing = var_smoothing
        self.class_priors = []
        self.class_counts = None
        self.class_nums = []
        # Map the class lables into indexes.
        self.classes_map = { }
        self.means = None
        self.varances = None
        # Number of training items.
        self.train_nums = None
        self.train_features = None
    def fit(self, X, y):
        """Fit model with datas."""
        self.train_nums = len(y)
        self.train_features = len(X[0])
        self.calculate_class_priors(y)
        self.calculate_means(X, y)
        self.calculate_varances(X,y)
    def predict(self,new_points):
        pass
    def calculate_class_priors(self, y):
        """Calculate the frequency of each category in the training set."""
        class_counter = Counter(y)
        self.class_counts = 0
        for lable, nums in class_counter.items():
            # Map the class lables into indexes.
            self.classes_map[lable] = self.class_counts
            self.class_counts += 1
            # Calculate the frequency of each category.
            self.class_priors.append(nums/self.train_nums)
            self.class_nums.append(nums)

    def calculate_means(self, X, y):
        # 中文：将means初始化为长度为类别数（索引对应类别映射的索引），元素为长度等于特征数，值为0的ndarray。
        # 再遍历整个训练集，每个训练项的特征值加在，对应类别的ndarray上。
        # 最后，对应类别的特征之和除以该类别的训练数据个数，得到每个类别中，各项特征的均值。
        self.means = [np.zeros(self.train_features) for _ in range(self.class_counts)]
        for idx in range(self.train_nums):
            self.means[self.classes_map[y[idx]]] += X(idx)
        self.means = [sums / nums for sums, nums in zip(self.means,self.class_nums)]
    def calculate_varances(self, X, y):
        pass
