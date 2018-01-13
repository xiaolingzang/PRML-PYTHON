import numpy as np
import matplotlib.pyplot as plt

from myprml.myfeatures import PolynomialFeatures
from myprml.linear import(
LeastSquaresClassifier,
LogisticRegressor,
SoftmaxRegressor,
LinearDiscriminantAnalyzer,
BayesianLogisticRegression

)
np.random.seed(1234)

def create_toy_data(add_outliers=False,add_class=False):
    x0=np.random.normal(size=50).reshape(-1,2)-1
    x1=np.random.normal(size=50).reshape(-1,2)+1
    if add_outliers:
        x_1=np.random.normal(size=10).reshape(-1,2)+np.array([5.0,10.])
        return np.concatenate([x0,x1,x_1]),np.concatenate([np.zeros(25), np.ones(30)]).astype(np.int)
    if add_class:
        x2=np.random.normal(size=50).reshape(-1,2)+3
        return np.concatenate([x0,x1,x2]),np.concatenate([np.zeros(25),np.ones(25),2+np.ones(25)]).astype(np.int)
    return np.concatenate([x0,x1]),np.concatenate([np.zeros(25),np.ones(25)]).astype(np.int)


x_train,y_train=create_toy_data()
x1_test,x2_test=np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
x_test=np.array([x1_test,x2_test]).reshape(2,-1).T
print(x_test.shape)

feature=PolynomialFeatures(1)
X_train=feature.transform(x_train)
X_test=feature.transform(x_test)

model=LeastSquaresClassifier()
model.fit(X_train,y_train)
y=model.classify(X_test)
plt.figure(1)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
plt.contourf(x1_test,x2_test,y.reshape(100,100),alpha=0.2,levels=np.linspace(0,1,3))
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.gca().set_aspect('equal',adjustable='box')
#plt.show()


x_train,y_train=create_toy_data(add_outliers=True)
x1_test, x2_test = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(-5, 10, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature = PolynomialFeatures(1)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

least_squares=LeastSquaresClassifier()
least_squares.fit(X_train,y_train)
y_ls=least_squares.classify(X_test)

logistic_regressor=LogisticRegressor()
logistic_regressor.fit(X_train,y_train)
y_lr=logistic_regressor.classify(X_test)
plt.figure(2)
plt.subplot(1,2,1)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
plt.contourf(x1_test,x2_test,y_ls.reshape(100,100),alpha=0.2,levels=np.linspace(0,1,3))
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.gca().set_aspect('equal',adjustable='box')
plt.title("least squares")
plt.subplot(1,2,2)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
plt.contourf(x1_test,x2_test,y_lr.reshape(100,100),alpha=0.2,levels=np.linspace(0,1,3))
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.gca().set_aspect('equal',adjustable='box')
plt.title("logistic regression")
#plt.show()


x_train, y_train = create_toy_data(add_class=True)
x1_test, x2_test = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(-5, 10, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature = PolynomialFeatures(1)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

least_squares=LeastSquaresClassifier()
least_squares.fit(X_train,y_train)
y_ls=least_squares.classify(X_test)

logistic_regressor=SoftmaxRegressor()
logistic_regressor.fit(X_train,y_train,max_iter=1000,learning_rate=0.01)
y_lr=logistic_regressor.classify(X_test)
plt.figure(3)
plt.subplot(1,2,1)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test,x2_test,y_ls.reshape(100,100),alpha=0.2, levels=np.array([0., 0.5, 1.5, 2.]))
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.gca().set_aspect('equal',adjustable='box')
plt.title("least squares")
plt.subplot(1,2,2)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
plt.contourf(x1_test,x2_test,y_lr.reshape(100,100),alpha=0.2,levels=np.array([0.,0.5,1.5,2.]))
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.gca().set_aspect('equal',adjustable='box')
plt.title("Softmax regression")
#plt.show()

x_train,y_train=create_toy_data()
x1_test, x2_test = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(-5, 10, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature=PolynomialFeatures(degree=1)
X_train=feature.transform(x_train)
X_test=feature.transform(x_test)

model=SoftmaxRegressor()
model.fit(X_train,y_train,max_iter=1000,learning_rate=0.01)
y=model.classify(X_test)
plt.figure(4)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test,x2_test,y.reshape(100,100),alpha=0.2,levels=np.array([0., 0.5, 1.5, 2.]))
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.gca().set_aspect('equal',adjustable='box')
#plt.show()


x_train, y_train = create_toy_data()
x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

feature = PolynomialFeatures(degree=1)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

model=BayesianLogisticRegression(alpha=1.)
model.fit(X_train,y_train,max_iter=1000)
y=model.proba(X_test)

plt.figure(5)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test,x2_test,y.reshape(100,100),levels=np.linspace(0, 1, 5),alpha=0.2)
plt.colorbar()
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.gca().set_aspect('equal',adjustable='box')
plt.show()
