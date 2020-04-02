import numpy as np 
import progressbar

from MachineLearning.utils.misc import bar_widgets
from MachineLearning.utils.data_manipulation import batch_iterator


class NeuralNetwork:
    """
    Neural Network. Deep Learning base model.
    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation_data: tuple
        A tuple containing validation data and labels (X, y)
    """
    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.error = {"training": [], "validation": []}
        self.loss_function = loss()
        self.progressbar = progrssor(widgets=bar_widgets)

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y": y}
    
    def set_trainable(self, trainable):
        """ Method which enables freezing of the weights of the network's layers. """
        for layer in self.layers:
            layer.trainable = trainable
    
    def add(self, layer):
        """ Method which adds a layer to the neural network """
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())
        
        if hasattr(layer, "initialize"):
            layer.initialize(optimizer=optimizer)
        
        self.layers.append(layer)