import numpy as np
from myprml.linear.classifer import Classifier

class LeastSquaresClassifier(Classifier):
    """
    Least Squares classifier model
    y=argmax_k X@W
    """
    def __init__(self,W=None):
        self.W=W
    def _fit(self,X,t):
        self._check_input(X)
        self._check_target(t)
        T=np.eye(int(np.max(t))+1)[t]
        self.W=np.linalg.pinv(X)@T     #(n_features,n_classes)

    def _classify(self,X):
        return np.argmax(X@self.W,axis=-1)
