from tensor import Warabitensor
import numpy as np
from typing import Dict
from typing import Callable as F 

class WarabiLayer:
    def __init__(self)->None:
        self.parameters: Dict[str, Warabitensor]={}
        self.gradients: Dict[str, Warabitensor]= {}
    def forward(self, inputs:Warabitensor)-> Warabitensor:
        raise NotImplementedError
    def backpropagate(self, gradients:Warabitensor)->Warabitensor:
        raise NotImplementedError


class Linear(WarabiLayer):
    def __init__(self, input_size:int, output_size:int)->None:
        super().__init__()
        self.parameters['a']=np.random.randn(input_size, output_size)
        self.parameters['b']=np.random.randn(input_size)
    
    def forward(self, inputs:Warabitensor)->Warabitensor:
        self.inputs=inputs
        return inputs @ self.parameters['a']+ self.parameters['b']
    def backpropagate(self, gradient:Warabitensor)->Warabitensor:
        self.gradients['b']=np.sum(gradient, axis=0)
        self.gradients['a']=self.inputs.T @ gradient
        return gradient @ self.parameters['a'].T 

class WarabiActivation(WarabiLayer):
    def __init__(self, f:F, f_derive:F):
        super().__init__()
        self.f=f
        self.f_derive=f_derive
    def forward(self, inputs:Warabitensor)-> Warabitensor:
        self.inputs=inputs
        return self.f(inputs)
    def backpropagate(self, gradient:Warabitensor)-> Warabitensor:
        self.gradient=gradient
        return self.f_derive(self.inputs)*self.gradient


def sigmoid(x:Warabitensor, y:float=0.5)->Warabitensor:
    return (1/2)*(1+np.tanh((y/2)*x))

def sigmoid_derive(x:Warabitensor, y:float=0.5)-> Warabitensor:
    func=sigmoid(x, y)
    return  y*func*(1-func)

def tanh(x:Warabitensor)->Warabitensor:
    return np.tanh(x)

def tanh_derive(x:Warabitensor)-> Warabitensor:
    func = tanh(x)
    return 1-func**2

class Sigmoid(WarabiActivation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_derive)

class Tanh(WarabiActivation):
    def __init__(self):
        super().__init__(tanh, tanh_derive)

    
