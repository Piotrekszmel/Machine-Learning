import math 
import numpy as np 
import copy 
from typing import Tuple

from MachineLearning.deep_learning.activation_functions import Sigmoid, ReLU


class Layer:

    def set_input_shape(self, shape):
        """ 
        Sets the shape that the layer expects of the input in the forward
        pass method 
        """
        self.input_shape = shape
    
    def layer_name(self):
        """ The name of the layer. """
        return self.__class__.__name__
    
    def parameters(self):
        """
        The number of trainable parameters used by layer
        """
        return 0
    
    def forward_pass(self, X, training):
        """ Forward propagation """
        raise NotImplementedError()

    def backward_pass(self, gradient):
        """ 
        it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer
        """
        raise NotImplementedError()

    def output_shape(self):
        """ 
        The shape of the output produces by forward pass
        """
        raise NotImplementedError()


class Dense(Layer):
    """
    Fully connected layer in neural network

    Parameters:
    -----------
    n_units (int) : Number of neurons in the layer
    input_shape (tuple) : The expected input shape of the layer. For dense layer
                          its a single number specifying the number of features of
                          the input. Must be specified if it is the first layer.
    """
    def __init__(self, n_units: int, input_shape: Tuple = None) -> None:
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None
    
    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))

        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)
    
    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)
    
    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, gradient):
        W = self.W

        if self.trainable:
            grad_w = self.layer_input.T.dot(gradient)
            grad_w0 = np.sum(gradient, axis=0, keepdims=True)

            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        gradient = gradient.dot(W.T)
        return gradient
    
    def output_shape(self):
        return (self.n_units, )


activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
}


class Activation(Layer):
    """
    A layer that applies an activation operation to the input.
    
    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """
    def __init(self, name):
        self.activation_name = name 
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return f"Activation ({self.activation_func.__class__.__name__})"
    
    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)
    
    def backward_pass(self, gradient):
        return gradient * self.activation_func.gradient(self.layer_input)
    
    def output_shape(self):
        return self.input_shape