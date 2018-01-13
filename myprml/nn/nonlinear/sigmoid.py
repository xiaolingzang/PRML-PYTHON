import numpy as np
from myprml.nn.tensor.constant import Constant
from myprml.nn.tensor.tensor import Tensor
from myprml.nn.function import Function


class Sigmoid(Function):
    """
    logistic sigmoid function
    y=1/(1+exp(-x))
    """

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.ouput = np.tanh(x.value * 0.5) * 0.5 + 0.5
        if isinstance(self.x, Constant):
            return Constant(self.ouput)
        return Tensor(self.ouput, function=self)

    def backward(self, delta):
        dx = self.ouput * (1 - self.ouput) * delta
        self.x.backward(dx)


def sigmoid(x):
    """
    logistic sigmoid function
    y = 1 / (1 + exp(-x))
    """
    return Sigmoid().forward(x)
