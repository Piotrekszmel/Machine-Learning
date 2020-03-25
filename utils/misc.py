import progressbar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np 

from data_operation import calculate_correlation_matrix
from data_operation import calculate_covariance_matrix
from data_manipulation import standardize

bar_widgets = [
    "Training: ", progressbar.Percentage(), " ", progressbar.Bar(marker="-", left="[", right="]"),
    " ", progressbar.ETA() 
]


class Plot:
    def __init__(self):
        self.cmap = plt.get_cmap("viridis")
    
    def _transform(self, X, dim):
        covariance = calculate_correlation_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        # Sort eigenvalues and eigenvector by largest eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:dim]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])(:, :dim)
        X_transformed = X.dot(eigenvectors)
        
        return X_transformed