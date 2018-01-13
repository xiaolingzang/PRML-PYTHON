import numpy as np
from myprml.nn.tensor.constant import Constant
from myprml.nn.tensor.tensor import Tensor
from myprml.nn.function import Function


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        x = self._convert2tensor(x)
        if self.axes is not None:
            self._equal_ndim(x, len(self.axes))
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(np.transpose(x.value, self.axes))
        return Tensor(np.transpose(x.value, self.axes), function=self)

    def backward(self, delta):
        if self.axes is None:
            dx = np.transpose(delta)
        else:
            dx = np.transpose(delta, np.argsort(self.axes))
        self.x.backward()


def transpose(x, axes=None):
    return Transpose(axes).forward(x)


def transpose_method(x, *args):
    if args == ():
        args = None
    return Transpose(args).forward(x)
