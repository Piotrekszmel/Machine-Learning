import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from regression import PolynomialRidgeRegression
from utils.data_manipulation import k_fold_cross_validation_sets, normalize, train_test_split, polynomial_features
from utils.data_operation import mean_squared_error
from utils.misc import Plot