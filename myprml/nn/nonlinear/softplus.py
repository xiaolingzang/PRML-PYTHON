import numpy as np
from myprml.nn.tensor.constant import Constant
from myprml.nn.tensor.tensor import Tensor
from myprml.nn.function import Function

class Softplus(Function):

    def forward(self,x):
        x=self._convert2tensor(x)
        self.x=x
        output=np.maximum(x.value,0)+np.log1p(np.exp(-np.abs(x.value))) #Calculates log(1 + x)
        if isinstance(x,Constant):
            return Constant(output)
        return Tensor(output,function=self)

    def backward(self,delta):
        dx=(np.tanh(0.5*self.x.value)*0.5+0.5)*delta
        self.x.backward(dx)

def softplus(x):
    """
    smoothed rectified linear unit
    log(1 + exp(x))
    """
    return Softplus().forward(x)