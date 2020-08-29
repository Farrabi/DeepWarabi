from tensor import Warabitensor
import numpy as np
from typing import Iterator, NamedTuple
Batch = NamedTuple('Batch', [('inputs', Warabitensor), ('targets', Warabitensor)])


class WarabiData:
    def __call__(self, inputs: Warabitensor, targets: Warabitensor)->Iterator[Batch]:
        raise NotImplementedError
    

class BatchIterator(WarabiData): 
    def __init__(self, batch_size:int =12, mazeru: bool=True)-> None:
        self.batch_size=batch_size
        self.mazeru=mazeru
    def __call__(self, inputs: Warabitensor, targets: Warabitensor)->Iterator[Batch]:
        begin= np.arange(0, len(inputs), self.batch_size)
        if self.mazeru:
            np.random.shuffle(begin)

        for i in begin:
            end = i + self.batch_size
            batch_inputs=inputs[i:end]
            batch_targets=targets[i:end]
            yield Batch(batch_inputs, batch_targets)

