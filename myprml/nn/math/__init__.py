from myprml.nn.math.add import  add
from myprml.nn.math.divide import divide,rdivide
from myprml.nn.math.exp import exp
from myprml.nn.math.log import log
from myprml.nn.math.matmul import matmul,rmatmul
from myprml.nn.math.mean import mean
from myprml.nn.math.multiply import multiply
from myprml.nn.math.negative import negative
from myprml.nn.math.power import power,rpower
from myprml.nn.math.product import prod
from myprml.nn.math.sqrt import sqrt
from myprml.nn.math.square import square
from myprml.nn.math.subtract import subtract,rsubtract
from myprml.nn.math.sum import sum

from myprml.nn.tensor import Tensor
Tensor.__add__ = add
Tensor.__radd__ = add
Tensor.__truediv__ = divide
Tensor.__rtruediv__ = rdivide
Tensor.mean = mean
Tensor.__matmul__ = matmul
Tensor.__rmatmul__ = rmatmul
Tensor.__mul__ = multiply
Tensor.__rmul__ = multiply
Tensor.__neg__ = negative
Tensor.__pow__ = power
Tensor.__rpow__ = rpower
Tensor.prod = prod
Tensor.__sub__ = subtract
Tensor.__rsub__ = rsubtract
Tensor.sum = sum


__all__ = [
    "add",
    "divide",
    "exp",
    "log",
    "matmul",
    "mean",
    "multiply",
    "power",
    "prod",
    "sqrt",
    "square",
    "subtract",
    "sum"
]
