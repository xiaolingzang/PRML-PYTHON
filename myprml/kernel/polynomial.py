import numpy as np
from myprml.kernel.kernel import Kernel

class PolynomialKernel(Kernel):
    """
    Polynomial kernel
    k(x,y) = (x @ y + c)^M
    """

    def __init__(self,degree=2,const=0.):
        """
        construct Polynomial kernel
        :param degree: int degree of polynomial order
        :param const: float  a constant to be added
        """
        self.const=const
        self.degree=degree

    def __call__(self,x,y,pairwise=True):
        """
        calculate pairwise polynomial kernel
        :param x: (..., ndim) ndarray
        :param y: (..., ndim) ndarray  another input with the same shape
        :param pairwise:
        :return:  polynomial kernel
        """
        if pairwise:
            x,y=self._pairwise(x,y)
        return (np.sum(x*y,axis=-1)+self.const)**self.degree