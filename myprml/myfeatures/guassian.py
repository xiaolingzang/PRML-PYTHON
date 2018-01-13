import numpy as np


class GaussianFeatures():
    """
    guassian function=exp(-0.5*(x-m)/v)

    """

    def __init__(self, mean, var):
        """
        construct guassian features

        :param mean: (n_features,ndim) or (n_features,) ndarray
        :param var: variance of the guassian function
        """
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
        assert isinstance(var, float) or isinstance(var, int)
        self.mean = mean
        self.var = var

    def _gauss(self, x, mean):
        return np.exp(-0.5 * np.sum(np.square(x - mean), axis=-1) / self.var)

    def transform(self, x):
        """
        transform input array with guassin features

        :param x: (sample_size,ndim) or (sample_size,)
        input array
        :return: (sample_size, n_features) guassian features

        """
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, 1) == np.size(self.mean, 1)
        basis = [np.ones(len(x))]
        for m in self.mean:
            basis.append(self._gauss(x, m))
        return np.asarray(basis).transpose()
