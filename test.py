import numpy as np
import Network
import Layer

units = [4, 5, 6]   
layers = [Layer.Layer(units[i], units[i + 1]) for i in range(len(units) - 1)]

nn = Network.Network(layers)


layer = Layer.Layer(input_dim=4, output_dim=2, activation=False)
print(layer.bias_derivative(np.array([[1, 2, 4, 5], [6, 7, 8, 9]]).reshape(2,4)))