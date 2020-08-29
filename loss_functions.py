from tensor import Warabitensor 

import numpy as np


class Loss:
    def loss(self, predicted:Warabitensor, current:Warabitensor)->float:
        raise NotImplementedError
    def grad(self, predicted:Warabitensor, current:Warabitensor)->Warabitensor:
        raise NotImplementedError

class MeanSquareWarabiError(Loss):
    def loss(self, predicted:Warabitensor, current:Warabitensor)->float:
        return np.sum(predicted-current)**2
    def grad(self, predicted:Warabitensor, current:Warabitensor)->Warabitensor:
        return 2*(predicted-current)


