import numpy as np
from matplotlib import pyplot as plt

def gramsschimidt(A):
    _, k = A.shape
    Q = A[:, [0]]/np.linalg.norm(A[:, 0])
    for j in range(1,k):
        q = A[:, j] - np.dot(Q, np.dot(Q.T, A[:, j]))
        nq = np.linalg.norm(q)
        if nq < 1e-9 * np.linalg.norm(A[:,j]):
            break
        Q = np.column_stack([Q, q / nq])
    return Q

for b in 2**np.mgrid[2:5]:
    A = np.random.normal(size=(b, b))
   
gram = gramsschimidt(A)

plt.figure()
plt.plot(gram, '+', label='gramsschimidt')
plt.legend(loc='best')
plt.show()