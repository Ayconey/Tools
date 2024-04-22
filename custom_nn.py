import numpy as np
from Addons import Addons


class DenseLayer:
    """
    Neural network layer that connects every node from previous layer to every of it's nodes.
    """
    def __init__(self, units, activation):
        self.units = units
        self.activation = Addons.get_activation_function(activation)

    def build(self,input_shape):
        """
         Method to initialize the layer.
        :param input_shape: Shape of an input from the previous layer ( previous_nodes x items )
        """

        # proper length for matrix multiplication ( last dimension is the number of items passed to the NN )
        length = input_shape[-1]

        # initialize random weights and biases
        self.w = np.random.rand(length, self.units)
        self.b = np.random.rand(self.units)

    def call(self,inputs):
        """
        Forward propagation.
        :param inputs: output from the previous layer
        :return: output of the layer
        """
        return self.activation(np.matmul(inputs, self.w) + self.b)

