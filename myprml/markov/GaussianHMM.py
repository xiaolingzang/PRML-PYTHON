import numpy as np
from myprml.rv import MultivariateGaussian
from myprml.markov.hmm import HiddenMarkovModel


class GaussianHMM(HiddenMarkovModel):
    """
    Hidden Markov Model with Gaussian emission model
    """

    def __init__(self, initial_proba, transition_proba, means, covs):

        """
          construct hidden markov model with Gaussian emission model
          :param initial_proba:
          :param transition_proba:
          :param means: (n_hidden, ndim) np.ndarray
                mean of each gaussian component
          :param covs:(n_hidden, ndim, ndim) np.ndarray
                 covariance matrix of each gaussian component

          Attributes
          ----------
          ndim : int
              dimensionality of observation space
          n_hidden : int
              number of hidden states
          """
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0] == \
               covs.shape[0]
        assert means.shape[1] == covs.shape[1] == covs.shape[2]
        super().__init__(initial_proba, transition_proba)
        self.ndim = means.shape[1]
        self.means=means
        self.covs = covs
        self.precisions = np.linalg.inv(self.covs)
        self.gaussians = [MultivariateGaussian(m, cov) for m, cov in zip(means, covs)]

    def draw(self, n=100):
        """
        draw random sequence from this model

        Parameters
        ----------
        n : int
            length of the random sequence

        Returns
        -------
        seq : (n, ndim) np.ndarray
            generated random sequence

        """
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)
        seq = []
        while len(seq) < n:
            seq.extend(self.gaussians[hidden_state].draw())
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state])
        return np.asarray(seq)

    def likelihood(self,X):
        diff = X[:, None, :] - self.means
        exponents = np.sum(
            np.einsum('nki,kij->nkj', diff, self.precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs) * (2 * np.pi) ** self.ndim)

    def maximize(self, seq, p_hidden, p_transition):
        self.initial_proba = p_hidden[0] / np.sum(p_hidden[0])
        self.transition_proba = np.sum(p_transition, axis=0) / np.sum(p_transition, axis=(0, 2))
        Nk = np.sum(p_hidden, axis=0)
        self.means = (seq.T @ p_hidden / Nk).T
        diffs = seq[:, None, :] - self.means
        self.covs = np.einsum('nki,nkj->kij', diffs, diffs * p_hidden[:, :, None]) / Nk[:, None, None]
