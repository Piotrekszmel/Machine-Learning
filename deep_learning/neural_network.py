from terminaltables import AsciiTable
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

    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)
        
        return layer_output

    def _backward_pass(self, loss_gradient):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        for layer in reversed(self.layers):
            loss_gradient = layer._backward_pass(loss_gradient)
    
    def train_on_batch(self, X, y):
        """ Single gradient update over one batch """
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        #gradient of loss funcyion w.r.t y_pred
        loss_gradient = self.loss_function.gradient(y, y_pred)
        #Backpropagate. Update weights
        self._backward_pass(loss_gradient)

        return loss, acc
    
    def test_on_batch(self, X, y):
        """ Evaluates the model over a single batch """
        y_pred self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc
    
    def fit(self, X, y, n_epochs, batch_size):
        """ Trains the model for a fixed number of epochs """
        for _ in self.progressbar(range(n_epochs)):

            batch_error = []
            for X_batch, y_batch in batch_iterator(x, y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X, y)
                batch_error.append(loss)
        
        self.errors["training"].append(np.mean(batch_error))

        if self.val_set is not None:
            val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
            self.errors["validations"].append(np.mean(val_loss))
        
        return self.errors["training"], self.errors["validation"]
    
    def summary(self, name="Model Sumarry"):
        print(AsciiTable([[name]]).table)
        print(f"Input Shape: {self.layers[0].input_shape}")
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        total_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            output_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(output_shape)])
            total_params += params

        print(AsciiTable(table_data).table)
        print(f"Total Parameters: {total_params}\n")
    
    def predict(self, X):
        """ Use the trained model to predict labels of X """
        return self._forward_pass(X, training=False)