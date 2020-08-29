from loss_functions import Loss, MeanSquareWarabiError
from optimizer import WarabiOptimizer, StochasticGradientDescent
from tensor import Warabitensor
from neuronal_net import WarabiNeuronal
from data import WarabiData, BatchIterator



def train(net:WarabiNeuronal,inputs: Warabitensor,targets: Warabitensor, 
          iterator:WarabiData =BatchIterator(),
          loss=MeanSquareWarabiError(),
          optimizer=StochasticGradientDescent(),num_epochs: int=1000)->None:
    for epoch in range(num_epochs):
        epoch_loss=0.0
        for batch in iterator(inputs, targets):
            predicted=net.forward(batch.inputs)
            epoch_loss+=loss.loss(predicted, batch.targets)
            gradient=loss.grad(predicted, batch.targets)
            net.backpropagate(gradient)
            optimizer.step(net)
            print(epoch, epoch_loss)








