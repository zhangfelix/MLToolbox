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
        self.class_priors = None
        self.class_counts = None
        self.classes = None
        self.means = None
        self.varances = None
    def fit(self, X, y):
        """Fit model with datas."""
        self.calculate_class_priors(y)
        self.calculate_means(X, y)
        self.calculate_varances(X,y)
    def predict(self,new_points):
        pass
    def calculate_class_priors(self, y):
        class_counter = Counter(y)
        for lable, nums in class_counter:
            pass

        pass
    def calculate_means(self, X, y):
        pass
    def calculate_varances(self, X, y):
        pass
