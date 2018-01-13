import numpy as np


class Classifier(object):
    """
    Base class for classifer
    """

    def fit(self, X, t, **kwargs):
        """
        eatimate parameters giver the training dataset

        :param X: (sample_size,n_features) np.ndarray
            trlain data input
        :param t: (sample_size,) np.ndarray
            train target
        :param kwargs:
        :return:
        """
        self._check_input(X)
        self._check_target(t)
        if hasattr(self, "_fit"):
            self._fit(X, t, **kwargs)
        else:
            raise NotImplementedError

    def classify(self, X, **kwargs):
        """
        classify inputs

        :param X: (sample_size,n_features) np.ndarray
            sample to classify
        :return: (sample_size,) np.ndarray
            labele index for each sample
        """
        if hasattr(self, "_classify"):
            return self._classify(X, **kwargs)
        else:
            raise NotImplementedError

    def proba(self, X, **kwargs):
        """
        compute probability of input belonging each class

        :param X: (sample_size,n_features) np.ndarray
            sample to compute their probability
        :param kwargs:
        :return: (sample_Szie,n_classes) np.ndarray
            probability for each class

        """
        self._check_input(X)
        if hasattr(self, "_proba"):
            return self._proba(X, **kwargs)
        else:
            raise NotImplementedError

    def _check_input(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X(input) must be np.ndarray")
        if X.ndim != 2:
            raise ValueError("X(input) must be two dimensional array")
        if hasattr(self, "n_features") and self.n_features != np.size(X, 1):
            raise ValueError(
                "mismatch in dimension 1 of X(input) (size {} is different from {})"
                    .format(np.size(X, 1), self.n_features)
            )

    def _check_target(self, t):
        if not isinstance(t, np.ndarray):
            raise ValueError("t(target) must be np.ndarray")
        if t.ndim != 1:
            raise ValueError("t(target) must be one dimensional array")
        if t.dtype != np.int:
            raise ValueError("dtype of t(target) must be np.int")
        if (t < 0).any():
            raise ValueError("t(target) must be only has 0 or 1")

    def _check_binary(self, t):
        if np.max(t) > 1 and np.min(t) < 0:
            raise ValueError("t(target) must only has 0 or 1")

    def _check_binary_negative(self, t):
        n_zeros = np.count_nonzero(t == 0)
        n_ones = np.count_nonzero(t == 1)
        if n_zeros + n_ones != t.size:
            raise ValueError("t(target) must only has -1 or 1")
