from myprml.nn.array.broadcast import broadcast_to
from myprml.nn.array.flatten import flatten
from myprml.nn.array.split import split
from myprml.nn.array.reshape import reshape, reshape_method
from myprml.nn.array.transpose import transpose, transpose_method
from myprml.nn.tensor.tensor import Tensor

Tensor.flatten = flatten
Tensor.reshape = reshape_method
Tensor.transpose = transpose_method

__all__ = [
    "broadcast_to",
    "flatten",
    "reshape",
    "split",
    "transpose"
]
