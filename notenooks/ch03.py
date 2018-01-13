import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
# %matplotlib inline

from myprml.myfeatures import GaussianFeatures, PolynomialFeatures,SigmoidalFeatures
from myprml.linear import (
    BayesianRegressor,
    EmpricalBayesRegressor,
    LinearRegressor,
    RidgeRegressor
)


# x = np.linspace(-1, 1, 100)
# X_polynomial = PolynomialFeatures(11).transform(x[:, None])
# print(X_polynomial.shape)
# X_gaussian = GaussianFeatures(np.linspace(-1, 1, 11), 0.1).transform(x)
# print(X_gaussian.shape)
# X_sigmoidal = SigmoidalFeatures(np.linspace(-1, 1, 11), 10).transform(x)
# print(X_sigmoidal.shape)
#
# plt.figure(figsize=(10, 5))
# for i, X in enumerate([X_polynomial, X_gaussian, X_sigmoidal]):
#     plt.subplot(1, 3, i + 1)
#     for j in range(12):
#         plt.plot(x, X[:, j])
# plt.show()


def create_toy_data(func,sample_size,std,domain=[0,1]):
    x=np.linspace(domain[0],domain[1],sample_size)
    np.random.shuffle(x)
    t=func(x)+np.random.normal(scale=std,size=x.shape)
    return x,t


def linear(x):
    return -0.3 + 0.5 * x

x_train, y_train = create_toy_data(linear, 20, 0.1, [-1, 1])
x = np.linspace(-1, 1, 100)
w0, w1 = np.meshgrid(
    np.linspace(-1, 1, 100),
    np.linspace(-1, 1, 100)
)
print(w0.shape)
print(w1.shape)
w = np.array([w0, w1]).transpose(1, 2, 0)
print(w.shape)

feature = PolynomialFeatures(degree=1)
X_train = feature.transform(x_train)
X = feature.transform(x)
model = BayesianRegressor(alpha=1., beta=100.)
for begin, end in [[0, 0], [0, 1], [1, 2], [2, 3], [3, 20]]:
    print(X_train[begin:end])
    print(y_train[begin:end])
    model.fit(X_train[begin:end], y_train[begin:end])
    plt.subplot(1, 2, 1)
    plt.scatter(-0.3, 0.5, s=200, marker="x")
    plt.contour(w0, w1, multivariate_normal.pdf(w, mean=model.w_mean, cov=model.w_cov))
    plt.gca().set_aspect('equal')
    plt.xlabel("$w_0$")
    plt.ylabel("$w_1$")
    plt.title("prior/posterior")

    plt.subplot(1, 2, 2)
    plt.scatter(x_train[:end], y_train[:end], s=100, facecolor="none", edgecolors="steelblue", lw=1)
    plt.plot(x,model.predict(X,sample_size=6),c="orange")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
