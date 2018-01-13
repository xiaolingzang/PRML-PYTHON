import random
import numpy as np


def rejection_sampling(func, rv, k, n):
    """
    perform rejection sampling n times

    :param func: callable
            (un)normalized distribution to be sampled from
    :param rv: distribution to generate sample
            distribution to generate sample
    :param k: float
             constant to be multiplied with the distribution
    :param n: int
            number of sample to draw
    :return: (n,ndim) ndarray
            generate sample

    """
    assert hasattr(rv, "draw"), "the distribution has no method to draw random samples"
    sample = []
    while len(sample) < n:
        sample_candidate = rv.draw()
        accept_proba = func(sample_candidate) / (k * rv.pdf(sample_candidate))
        if random.random() < accept_proba:
            sample.append(sample_candidate[0])
    sample = np.asarray(sample)
    assert sample.shape == (n, rv.ndim), sample.shape
    return sample
