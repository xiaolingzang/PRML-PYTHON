import numpy as np

class Kernel(object):
    """
    Base class for kernel function
    """

    def _pairwise(self,x,y):
        """
        all pairs of x and y
        :param x: (sample_size, n_features)   input
        :param y: (sample_size, n_features)   another input
        :return:  two array with shape (sample_size, sample_size, n_features)
        """
        return (
            np.tile(x,(len(y),1,1)).transpose(1,0,2),
            np.tile(y,(len(x),1,1))
        )

    # numpy.tile(A, reps)
    # Construct an array by repeating A the number of times given by reps.