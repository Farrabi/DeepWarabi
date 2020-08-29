from tensor import Warabitensor 
from layer import WarabiLayer
from typing import Sequence, Iterator, Tuple

class WarabiNeuronal:
    def __init__(self, layers:Sequence[WarabiLayer])-> None:
        self.layers=layers
    
    def forward(self, inputs:Warabitensor)->Warabitensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, gradient:Warabitensor)->Warabitensor:
        self.gradient=gradient
        for layer in reversed(self.layers):
            gradient = layer.backpropagate(gradient)
        return gradient
    def parameters_gradients(self)->Iterator[Tuple[Warabitensor, Warabitensor]]:
        for layer in self.layers:
            for name, parameter in layer.parameters.items():
                gradient=layer.gradients[name]
                yield parameter, gradient


