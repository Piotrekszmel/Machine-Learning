import numpy as np 

from MachineLearning.utils.data_manipulation import train_test_split, standardize, divide_on_feature
from MachineLearning.utils.data_operation import calculate_entropy, accuracy_score, calculate_variance


class DecisionNode:
    """
    Class that represents a decision node or leaf in the decision tree
    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """  
    def __init__(self, feature_i=None, threshold=None, value=None,
                true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch