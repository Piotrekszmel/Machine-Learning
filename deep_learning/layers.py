import math 
import numpy as np 
import copy 


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

    def backward_pass(self, grad):
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
    