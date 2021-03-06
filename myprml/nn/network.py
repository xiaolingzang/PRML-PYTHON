from myprml.nn.random.random import RandomVariable
from myprml.nn.tensor.parameter import Parameter


class Network(object):
    """
        a base class for network building

        Parameters
        ----------
        kwargs : tensor_like
            parameters to be optimized

        Attributes
        ----------
        parameter : dict
            dictionary of parameters to be optimized
        random_variable : dict
            dictionary of random varibles
    """

    def __init__(self, **kwargs):
        self.random_variable = {}
        self.parameter = {}
        for key, value in kwargs.items():
            if isinstance(value, Parameter):
                self.parameter[key] = value
            else:
                try:
                    value = Parameter(value)
                except TypeError:
                    raise TypeError(f"invalid type argument:{type(value)}")
                self.parameter[key] = value
            object.__setattr__(self, key, value)

    def __setattr__(self, key, value):
        if isinstance(value, RandomVariable):
            self.random_varibale[key] = value
        object.__setattr__(self, key, value)

    def clear(self):
        """
        clear gradient and constructed bayesian network
        :return:
        """
        for p in self.parameter.values():
            p.cleargrad()
        self.random_varibale = {}

    def log_pdf(self, coef=1):
        """
        compute logarithm of probabilty density function
        :param coef: float
            coefficient to balance likelihood and prior
            assuming mini-batch size / whole data size for mini-batch training
        :return:
        logp : tensor_like
            logarithm of probability density function
        """
        logp = 0
        for rv in self.random_varibale.values():
            if rv.observed:
                logp += rv.log_pdf().sum()
            else:
                logp += coef * rv.log_pdf().sum()
        return logp

    def elbo(self, coef=1.):
        """
         ompute evidence lower bound of this model
         ln p(output) >= elbo
        :param coef:
             coefficient to balance likelihood and prior
             assuming mini-batch size / whole data size for mini-batch training
        :return: evidence : tensor_like
             evidence lower bound
        """
        evidence = 0
        for rv in self.random_varibale.values():
            if rv.observed:
                evidence += rv.log_pdf().sum()
            else:
                evidence += -coef * rv.KLqp().sum()
        return evidence
