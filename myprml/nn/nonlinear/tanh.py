import numpy as np
from myprml.nn.tensor.constant import Constant
from myprml.nn.tensor.tensor import Tensor
from myprml.nn.function import Function


class Tanh(Function):
    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.tanh(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = (1 - np.square(self.output)) * delta  # 1-f(z)^2
        self.x.backward(dx)


def tanh(x):
    """
    hyperbolic tangent function
    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return Tanh().forward(x)
