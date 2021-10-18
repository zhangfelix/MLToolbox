"""
Navie bayes algorithms.
"""

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
        self.train_features = len(x[0])
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

    def calculate_means(self, X, y):
        self.means = [[] for _ in range(self.train_nums)]
        for idx in self.train_nums:
            self.means[self.classes_map[y[idx]]]
    def calculate_varances(self, X, y):
        pass
