from typing import Any
import numpy as np

def dxtanh(x):
    return 1 - np.tanh(x) ** 2

class Layer:
    # Layer function with tanh activation.
    # Follows the convention of representing vectors as rows-vectors.
    # Supports input data as a matrix in which each row represents a data point.
    def __init__(self, input_dim, output_dim, activation=True):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.zeros(input_dim * output_dim).reshape(input_dim, output_dim)
        self.bias = np.zeros(output_dim).reshape(1, output_dim)
        self.activation = activation

        
                                                                         
    def forward(self, x):
        if self.activation:
            return np.tanh(np.matmul(x, self.weight) + self.bias)
        else:
            return np.matmul(x, self.weight) + self.bias
        
    def input_derivative(self, x):
        if self.activation:
            aux = dxtanh(np.matmul(x, self.weight) + self.bias)

            # This line could be optimized
            aux = [np.diag(col) for col in aux]
            
            return np.matmul(self.weight, aux)
        else:
            return np.array([self.weight] * len(x))
        
    def weight_derivative(self, x):
        if self.activation:
            return np.array([[i * np.diag(dxtanh(np.matmul(row, self.weight) + self.bias)) for i in row] for row in x])
        else:
            identity = np.identity(self.output_dim)
            return np.array([[i * identity for i in row] for row in x])

    def bias_derivative(self, x):
        if self.activation:
            return np.array([np.diag(dxtanh(np.matmul(row, self.weight) + self.bias)) for row in x])
        else:
            return np.array([np.identity(self.output_dim)] * len(x))
        
    def get_weight(self):
        return self.weight
    
    def get_bias(self):
        return self.bias