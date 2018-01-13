import numpy as np

class RandomVariable(object):
    """
    base class for random varibale
    """

    def __init__(self):
        self.parameter={}

    def __repr__(self):
        string=f"{self.__class__.__name__}(\n"
        for key,value in self.parameter.items():
            string +=(" "*4)
            if isinstance(value,RandomVariable):
                string +=f"{key}={value:8}"
            else:
                string +=f"{key}={value}"
            string +="\n"
            string +=")"
            return string

    def __format__(self, indent="4"):
        indent=int(indent)
        string=f"{self.__class__.__name__}(\n"
        for key,value in self.parameter.items():
            string +=(" "*indent)
            if isinstance(value,RandomVariable):
                string +=f"{key}="+value.__format__(str(indent+4))
            else:
                string +=f"{key}={value}"
            string+="\n"
        string +=(" "*(indent-4))+" "
        return string
    def fit(self,X,**kwargs):
        """
        estimate parameters of the distribution

        :param X: np.ndarray   observed data
        :param kwargs:
        :return:
        """

        self._check_input(X)
        if hasattr(self,"_fit"):
            self._fit(X,**kwargs)
        else:
            raise NotImplementedError

    def pdf(self,X):
        """
        compute probability demsity function
        :param X: (sample_size, ndim) np.ndarray
        :return:  value of probability density function for each input
        """
        self._check_input(X)
        # print(hasattr(self, "_pdf"))
        if hasattr(self, "_pdf"):
            return self._pdf(X)
        else:
            raise NotImplementedError

    def draw(self,sample_size=1):
        """
        draw sample from the distribution
        :param sample_size:
        :return:
        """
        assert isinstance(sample_size,int)
        if hasattr(self,"_draw"):
            return self._draw(sample_size)
        else:
            raise NotImplementedError

    def _check_input(self,X):
        assert isinstance(X,np.ndarray)