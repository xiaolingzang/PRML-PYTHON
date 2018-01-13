import numpy as np
from scipy.special import gamma
from myprml.rv.rv import RandomVariable

np.seterr(all="ignore")

class Gamma(RandomVariable):
    """
    Gamma distribution
    p(x|a,b)=b^a x^(a-1) exp(-bx)/gamma(a)

    """

    def __init__(self,a,b):
        """
        constuct Gamma distribution

        :param a: int, float, or np.ndarray
                 shape parameter
        :param b: int, float, or np.ndarray
                 arte paramater
        """
        super().__init__()
        a=np.asarray(a)
        b=np.asarray(b)
        assert a.shape==b.shape
        self.a=a
        self.b=b

    @property
    def a(self):
        return self.parameter["a"]
    @a.setter
    def a(self,a):
        if isinstance(a,(int,float,np.number)):
            if a<=0:
                raise ValueError("a must be positive")
            self.parameter["a"]=np.asarray(a)
        elif isinstance(a,np.ndarray):
            if (a<0).any():
                raise ValueError("a must be positive")
            self.parameter["a"]=a
        else:
            if a is not None:
                raise TypeError(f"{type(a)} is not supported for a")
            self.parameter["a"]=None

    @property
    def b(self):
        return self.parameter["b"]

    @b.setter
    def b(self,b):
        if isinstance(b,(int,float,np.number)):
            if b<=0:
                raise ValueError("b must be positive")
            self.parameter["b"]=np.asarray(b)
        elif isinstance(b,np.ndarray):
            if (b<0).any():
                raise ValueError("a must be positive")
            self.parameter["b"]=b
        else:
            if b is not None:
                raise TypeError(f"{type(b)} is not supported for a")
            self.parameter["b"]=None

    @property
    def ndim(self):
        return self.a.dim

    @property
    def shape(self):
        return self.a.shape

    @property
    def size(self):
        return self.a.size

    def _pdf(self,X):
        return (
            self.b**self.a
            *X**(self.a-1)
            *np.exp(-self.b*X)
            /gamma(self.a))

    def _draw(self,sample_size=1):
        return np.random.gamma(
            shape=self.a,
            scale=1/self.b,
            size=(sample_size,)+self.shape
        )

# def myfun():
#     prop=Gamma(a=1,b=2)
#     print(prop.draw(10))
#
# myfun()