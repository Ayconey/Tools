import numpy as np


# decorator to make the whole class static
def static_class(cls):
    for name, method in vars(cls).items():
        if callable(method):
            setattr(cls, name, staticmethod(method))
    return cls

class Addons:
    """
    Class that contains some useful functions.
    """
    def __init__(self):
        pass
    @staticmethod
    def get_activation_function(activation):
        """
        Method to get activation function from string.
        :param activation: name of the activation function
        :return: proper math function
        """
        activation_functions = {
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'softmax': lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0),
            'linear': lambda x: x,
            'swish': lambda x: x * (1 / (1 + np.exp(-x))),
            'leaky_relu': lambda x: np.maximum(0.01 * x, x),
            'elu': lambda x: x if x >= 0 else 0.01 * (np.exp(x) - 1)
        }
        if activation in activation_functions:
            return activation_functions[activation]
        else:
            raise ValueError(f"There is no such activation function as: {activation}")