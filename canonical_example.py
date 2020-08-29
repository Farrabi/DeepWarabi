import numpy as np 

from training import  train
from neuronal_net import WarabiNeuronal
from layer import Linear, Sigmoid, Tanh

inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ])

targets= np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
        ]) 

net = WarabiNeuronal([
    Linear(input_size=2, output_size=2), 
    Tanh(),
    Linear(input_size=2, output_size=2)
    ])

train(net, inputs, targets)

for i, j in zip(inputs, targets):
    predicted=net.forward(i)
    print(i, predicted, j)

