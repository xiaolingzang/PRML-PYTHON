import matplotlib.pyplot as plt
import numpy as np

from myprml.rv import Gaussian,Uniform
from myprml.sampling import(
metropolis,
metropolis_hastings,
rejection_sampling,
sir
)
np.random.seed(1234)


def func(x):
    return np.exp(-x**2)+3*np.exp(-(x-3)**2)

x=np.linspace(-5,10,100)

plt.figure(1)
rv=Gaussian(mu=np.array([2.]),var=np.array([2.]))
plt.plot(x,func(x),label=r"$\tilde{p}(z)$")
plt.plot(x,15*rv.pdf(x),label=r"$kq(z)$")
plt.legend(fontsize=15)
plt.show()

plt.figure(2)
samples=rejection_sampling(func,rv,k=15,n=100)
plt.plot(x,func(x),label=r"$\tilde{p}(z)$")
plt.hist(samples,normed=True,alpha=0.2)
plt.scatter(samples,np.random.normal(scale=0.03,size=(100,1)),s=5,label="sample")
plt.legend(fontsize=15)
plt.show()


plt.figure(3)
samples=sir(func,rv,n=100)
plt.plot(x,func(x),label=r"$\tilde{p}(z)$")
plt.hist(samples,normed=True,alpha=0.2)
plt.scatter(samples,np.random.normal(scale=0.03,size=(100,1)),s=5,label="sample")
plt.legend(fontsize=15)
plt.show()

plt.figure(4)
samples=metropolis(func,Gaussian(mu=np.zeros(1),var=np.ones(1)),n=100,downsample=10)
plt.plot(x,func(x),label=r"$\tilde{p}(z)$")
plt.hist(samples,normed=True,alpha=0.2)
plt.scatter(samples,np.random.normal(scale=0.03,size=(100,1)),s=5,label="sample")
plt.legend(fontsize=15)
plt.show()

plt.figure(5)
samples=metropolis_hastings(func,Gaussian(mu=np.ones(1),var=np.ones(1)),n=100,downsample=10)
plt.plot(x,func(x),label=r"$\tilde{p}(z)$")
plt.hist(samples,normed=True,alpha=0.2)
plt.scatter(samples,np.random.normal(scale=0.03,size=(100,1)),s=5,label="sample")
plt.legend(fontsize=15)
plt.show()
