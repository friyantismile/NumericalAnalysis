import numpy as np

def kronmultv(A, B, x):
    n, k = A.shape[1], B.shape[1]
    assert x.size == n * k, 'size mismatch'
    xx = np.reshape(x, (n, k))
    Z = np.dot(xx, B.T)
    yy=np.dot(A,Z)
    return np.ravel(yy)