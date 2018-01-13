import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from myprml.rv import (
Bernoulli,
Beta,
Categorical,
Dirichlet,
Gamma,
uniform,
Gaussian,
StudentsT,
MultivariateGaussian,
MultivariateGaussianMixture
)
np.random.seed(1234)

# x=np.linspace(0,1,100)
# a=0.1
# b=0.1
# beta=Beta(a,b)
# beta.draw(10)

# for i,[a,b] in enumerate([[0.1,0.1],[1,1],[2,3],[8,4]]):
#     plt.plot(2,2,i+1)
#     beta=Beta(a,b)
#     plt.xlim(0,1)
#     plt.ylim(0,3)
#     plt.plot(x,beta.pdf(x))
#     plt.annotate("a={}".format(a),(0.1,2.5))
#     plt.annotate("b={}".format(b),(0.1,2.1))
# plt.show()

#
# mu=Gaussian(0,0.1)
# model=Gaussian(mu,0.1)
# x=np.linspace(-1,1,100)
# plt.plot(x,model.mu.pdf(x),label="N=0")
# model.fit(np.random.normal(loc=0.8,scale=0.1,size=1))
# plt.plot(x,model.mu.pdf(x),label="N=1")
#
# model.fit(np.random.normal(loc=0.8, scale=0.1, size=1))
# plt.plot(x,model.mu.pdf(x),label="N=2")
#
# model.fit(np.random.normal(loc=0.8, scale=0.1, size=8))
# plt.plot(x,model.mu.pdf(x),label="N=10")
#
# plt.xlim(-1,1)
# plt.ylim(0,5)
# plt.legend()
# plt.show()


# x=np.linspace(0,2,100)
# for i,[a,b] in enumerate([[0.1, 0.1], [1, 1], [2, 3], [4, 6]]):
#     plt.subplot(2,2,i+1)
#     gamma=Gamma(a,b)
#     plt.xlim(0,2)
#     plt.ylim(0,2)
#     plt.plot(x,gamma.pdf(x))
#     plt.annotate("a={}".format(a),(1,1.6))
#     plt.annotate("b={}".format(b),(1,1.3))
# plt.show()
#
# X=np.random.normal(size=20)
# X=np.concatenate([X, np.random.normal(loc=20., size=3)])
# plt.hist(X.ravel(),bins=50,normed=1.,label="samples")
#
# students_t=StudentsT()
# gaussian=Gaussian()
# gaussian.fit(X)
# students_t.fit(X)
#
# print(gaussian)
# print(students_t)
#
# x=np.linspace(-5,25,1000)
# plt.plot(x,students_t.pdf(x),label="student's t",linewidth=2)
# plt.plot(x, gaussian.pdf(x), label="gaussian", linewidth=2)
# plt.legend()
# plt.show()

# # -*- coding: utf-8 -*-
# """
# Created on Mon Nov 13 15:44:30 2017
#
# @author: zangxiaoling
# """
#
# import numpy as np
# from scipy.spatial.distance import cdist
#
# x1 = np.random.normal(size=(100, 2))
# x1 += np.array([-5, 5])
# x2 = np.random.normal(size=(100, 2))
# x2 += np.array([-5, 5])
# x3 = np.random.normal(size=(100, 2))
# x3 += np.array([0, 5])
# X = np.vstack((x1, x2, x3))
#
#
# class KMeans():
#
#     def __init__(self, n_clusters):
#         self.n_clusters = n_clusters
#
#     def fit(self, X, iter_max=100):
#         """
#         performe k-means algorithm
#
#         :param X: (sample_size,n_features) ndarray
#         input data
#         :param iter_max: int
#         maximum number of iterations
#
#         :return: (n_clusters,n_features) ndarray
#         center of each cluster
#
#         """
#         I = np.eye(self.n_clusters)
#         centers = X[np.random.choice(len(X), self.n_clusters, replace=False)]
#         for _ in range(iter_max):
#             prev_centers = np.copy(centers)
#             D = cdist(X, centers)
#             cluster_index = np.argmin(D, axis=1)
#             cluster_index = I[cluster_index]
#             centers = np.sum(X[:, None, :] * cluster_index[:, :, None], axis=0)/ np.sum(cluster_index, axis=0)[:, None]
#             if np.allclose(prev_centers, centers):
#                 break
#             self.centers = centers
#
#     def predict(self, X):
#         """
#         calculate closest cluster center index
#
#         :param X: (sample_size,n_features)
#         input data
#         :return: (sample_size,) ndarray
#         indcates with cluster they belong
#
#         """
#         D = cdist(X, self.centers)
#         return np.argmin(D, axis=1)
#
#
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)

x1=np.random.normal(size=(100,2))
x1 +=np.array([-5,5])
x2=np.random.normal(size=(100,2))
x2 +=np.array([-5,5])
x3=np.random.normal(size=(100,2))
x3 +=np.array([0,5])
X=np.vstack((x1,x2,x3))
model = MultivariateGaussianMixture(n_components=3)
model.fit(X)
print(model)

x_test,y_test=np.meshgrid(np.linspace(-10,10,100),np.linspace(-10,10,100))
X_test=np.array([x_test,y_test]).reshape(2,-1).transpose()
probs=model.pdf(X_test)
Probs=probs.reshape(100,100)
plt.scatter(X[:,0],X[:,1])
plt.contour(x_test,y_test,Probs)
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.show()