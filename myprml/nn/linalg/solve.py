import numpy as np
from myprml.nn.tensor.constant import Constant
from myprml.nn.tensor.tensor import Tensor
from myprml.nn.function import Function


class Solve(Function):
    def forward(self, a, b):
        a = self._convert2tensor(a)
        b = self._convert2tensor(b)
        self._equal_ndim(a, 2)
        self._equal_ndim(b, 2)
        self.a = a
        self.b = b
        self.ouput = np.linalg.solve(a.value, b.value)
        if isinstance(self.a, Constant) and isinstance(self.b, Constant):
            return Constant(self.ouput)
        return Tensor(self.ouput, function=self)

    def backward(self, delta):
        db = np.linalg.solve(self.a.value.T, delta)
        da = -db @ self.ouput.T
        self.a.backward(da)
        self.b.backward(db)


def solve(a, b):
    """
    solve a linear matrix equation
    ax = b
    :param a: (d, d) tensor_like     coefficient matrix
    :param b: (d, k) tensor_like    dependent variable
    :return: (d, k) tensor_like      solution of the equation
    """
    return Solve.forward(a, b)
