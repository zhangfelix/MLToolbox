"""
Navie bayes algorithms.
"""

from collections import Counter
class GaussianNB:
    """
    GaussianNB Class.
    """
    def __init__(self, var_smoothing=1e-09):
        self.var_smoothing = var_smoothing
        self.class_priors = []
        self.class_counts = None
        self.classes_map = { }
        self.means = None
        self.varances = None
        self.train_nums = None
    def fit(self, X, y):
        """Fit model with datas."""
        self.train_nums = len(y)
        self.train_features = len(x[0])
        self.calculate_class_priors(y)
        self.calculate_means(X, y)
        self.calculate_varances(X,y)
    def predict(self,new_points):
        pass
    def calculate_class_priors(self, y):
        class_counter = Counter(y)
        self.class_counts = 0
        for lable, nums in class_counter.items():
            self.classes_map[lable] = self.class_counts
            self.class_counts += 1
            self.class_priors.append(nums/self.train_nums)

    def calculate_means(self, X, y):
        self.means = [[] for _ in range(self.train_nums)]
        for idx in self.train_nums:
            self.means[self.classes_map[y[idx]]]
    def calculate_varances(self, X, y):
        pass
