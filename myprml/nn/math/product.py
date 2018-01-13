import numpy as np
from myprml.nn.tensor.constant import Constant
from myprml.nn.tensor.tensor import Tensor
from myprml.nn.function import Function


class Product(Function):

    def __init__(self, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis,)
        elif isinstance(axis, tuple):
            axis = tuple(sorted(axis))
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.prod(self.x.value, axis=self.axis, keepdims=True)
        if not self.keepdims:
            output = np.squeeze(self.output)  #np.squeeze：Remove single-dimensional entries from the shape of an array.
            if output.size == 1:
                output = output.item()
        else:
            output = self.output
        if isinstance(self.x, Constant):
            return Constant(output)
        return Tensor(output, function=self)

    def backward(self, delta):
        if not self.keepdims and self.axis is not None:
            for ax in self.axis:
                delta = np.expand_dims(delta, ax)
        dx = delta * self.output / self.x.value
        self.x.backward(dx)


def prod(x, axis=None, keepdims=False):
    """
    product of all element in the array
    Parameters
    ----------
    x : tensor_like
        input array
    axis : int, tuple of ints
        axis or axes along which a product is performed
    keepdims : bool
        keep dimensionality or not
    Returns
    -------
    product : tensor_like
        product of all element
    """
    return Product(axis=axis, keepdims=keepdims).forward(x)
