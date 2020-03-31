import numpy as np 
from typing import List

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


class DecisionTree:
    """
    Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """
    def __init__(self, min_samples_split: int = 2, min_impurity: float = 1e-7, 
                max_depth: int = float("inf"), loss=None) -> None:
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None
        self._leaf_value_calculation = None
        self.one_dim = None
        self.loss = loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, loss=None) -> None:
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, current_depth: int = 0) -> DecisionNode:
        """ 
        Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data
        """

        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self._impurity_calculation(y, y1, y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                                }

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)
    
    def predict_value(self, X: np.ndarray, tree: DecisionNode = None) -> np.float64:
        """ 
        Do a recursive search down the tree and make a prediction of the data sample by the
        value of the leaf that we end up at 
        """
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value
        
        feature_value = X[tree.feature_i]
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        return self.predict_value(X, branch)
    
    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred
    
    def print_tree(self, tree: DecisionNode = None, indent=" ") -> None:
        """ Recursively print the decision tree """
        if not tree:
            self.tree = self.root
        
        if tree.value is not None:
            print(tree.value)
        
        else:
            print(f"{tree.feature_values: tree.threshold}")
            print(f"{indent}T->", end="")
            self.print_tree(tree.true_branch, index + indent)
            print(f"{index}F->", end="")
            self.print_tree(tree.false_branch, index + indent)


class RegressionTree(DecisionTree):
    def _calculate_variance_reduction(self, y: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.float64:
        var_total = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        variance_reduction = var_total - (var_1 * frac_1 + var_2 * frac_2)
        return sum(variance_reduction)
    
    def _mean_of_y(self, y: np.ndarray) -> np.ndarray:
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super().fit(X, y)


class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

        return info_gain
    
    def _majority_vote(self, y: np.ndarray) -> int:
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = label
        return most_common
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super().fit(X, y)