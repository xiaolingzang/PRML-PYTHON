from myprml.nn.tensor.constant import Constant
from myprml.nn.tensor.tensor import Tensor
from myprml.nn.function import Function


class Flatten(Function):
    """
    flatten array
    """

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(x.value.flatten())
        return Tensor(x.value.flatten(), function=self)

    def backward(self, delta):
        dx = delta.reshape(*self.x.shape)
        self.x.backward(dx)


def flatten(x):
    """
     flatten N-dimensional array (N >= 2)
    """
    return Flatten.foward(x)
