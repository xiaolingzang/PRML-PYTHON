import numpy as np
from myprml.nn.tensor.constant import Constant
from myprml.nn.tensor.tensor import Tensor
from myprml.nn.function import Function


class Inverse(Function):
    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self._equal_ndim(x, 2)
        self.ouput = np.linalg.inv(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.ouput)
        return Tensor(self.ouput, function=self)

    def backward(self, delta):
        dx = -self.ouput.T @ delta @ self.ouput.T
        self.x.backward(dx)


def inv(x):
    """
        inverse of a matrix
    :param x: (d, d) tensor_like   a matrix to be inverted
    :return: (d, d) tensor_like    inverse of the input
    """
    return Inverse.forward(x)
