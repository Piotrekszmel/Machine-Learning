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
    
    def plot_regression(self, lines, title, axis_labels=None, mse=None, scatter=None, legend={"type": "lines", "loc": "lower right"})
        if scatter: 
            scatter_plots = scatter_labels = []
            for s in scatter:
                scatter_plots += [plt.scatter(s["x"], s["y"], color=s["color"], s=s["size"])]
                scatter_labels += [s["label"]]
            scatter_plots = tuple(scatter_plots)
            scatter_labels = tuple(scatter_labels)
        
        for l in lines:
            li = plt.plt(l["x"], l["y"], color=l["color"], linewidth=l["width"], label=l["label"])
        
        if mse:
            plt.suptitle(title)
            plt.title(f"MSE: {mse:.2f}", fontsize=10)
        else:
            plt.title(title)
        
        if axis_labels: 
            plt.xlabel(axis_labels["x"])
            plt.ylabel(axis_labels["y"])
        
        if legend["type"] == "lines":
            plt.legend(loc="lower left")
        elif legend["type"] == "scatter" and scatter:
            plt.legend(scatter_plots, scatter_labels, loc=legend["loc"])
        
        plt.show()