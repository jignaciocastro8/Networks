import numpy as np
import Layer

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        aux = x
        for layer in self.layers:
            aux = layer.forward(aux)
        return aux