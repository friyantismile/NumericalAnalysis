import numpy as np
from matplotlib import pyplot as plt

def kronmultv(A, B, x):
    n, k = A.shape[1], B.shape[1]
    assert x.size == n * k, 'size mismatch'
    xx = np.reshape(x, (n, k))
    Z = np.dot(xx, B.T)
    yy=np.dot(A,Z)
    return np.ravel(yy)

for b in 2**np.mgrid[2:5]:
    vector = np.random.normal(size=(b, b)) #vectorXd d=VectordXD::random(n,1)
    A = np.random.normal(size=(b, b))
    B = np.random.normal(size=(b, b))

kron = kronmultv(A, B, vector)
plt.figure()
plt.plot(kron, '+', label='Kronmultv')
plt.legend(loc='best')
plt.show()