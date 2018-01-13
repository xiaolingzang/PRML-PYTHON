import numpy as np

class SupportVectorClassifier(object):

    def __init__(self,kernel,C=np.Inf):
        """
        construct support vector classifier
        :param kernel: kernel function to compute inner products
        :param C: penalty of misclassification
        """
        self.kernel=kernel
        self.C=C

    def fit(self,X,t,learning_rate=0.1,decay_step=10000,decay_rate=0.9,min_lr=1e-5):
        """
        estimate decision boundary and its support vectors
        :param X: (sample_size, n_features) ndarray  input datd
        :param t: (sample_size,) ndarray   corresponding labels 1 or -1
        :param learning_rate: update ratio of the lagrange multiplier
        :param decay_step: number of iterations till decay
        :param decay_rate: rate of learning rate decay
        :param min_lr: minimum value of learning rate
        :return:  (n_vector, n_features) ndarray     support vectors of the boundary

                Attributes
        ----------
        a : (sample_size,) ndarray
            lagrange multiplier
        b : float
            bias parameter
        """

        if X.ndim==1:
            X=X[:,None]
        assert X.ndim==2
        assert t.ndim==1
        lr=learning_rate
        t2=np.sum(np.square(t))
        if self.C==np.Inf:
            a=np.ones(len(t))
        else:
            a=np.zeros(len(t))+self.C/10
        Gram=self.kernel(X,X)
        H=t*t[:,None]*Gram    #t*t[:,None] 大小是len(t)*len(t)
        while True:
            for i in range(decay_step):
                grad=1-H@a
                a+=lr*grad
                a-=(a@t)*t/t2
                np.clip(a,0,self.C,out=a)
            mask=a>0
            self.X=X[mask]
            self.t=t[mask]
            self.a=a[mask]
            self.b=np.mean(
                self.t-np.sum(
                    self.a*self.t*self.kernel(self.X,self.X),axis=-1
                )
            )
            if self.C==np.Inf:
                if np.allclose(self.distance(self.X)*self.t,1,rtol=0.01,atol=0.01):
                    break
            else:
                if np.all(np.greater_equal(1.01,self.distance(self.X)*self.t)):
                    break
            if lr<min_lr:
                break
            lr*=decay_rate

    def predict(self,x):
        """
        predict labels of the input

        :param x: (sample_size, n_features) ndarray   input
        :return: (sample_size,) ndarray    predicted labels
        y=self.distance(x)
        label=np.sign(y)
        """
        y=self.distance(x)
        label=np.sign(y)
        return label
    def distance(self,x):
        """
        calculate distance from the decision boundary
        :param x: (sample_size, n_features) ndarray   input
        :return: distance from the boundary
        """
        distance=np.sum(self.a*self.t*self.kernel(x,self.X),axis=-1)+self.b
        return distance