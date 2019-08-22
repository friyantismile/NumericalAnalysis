import numpy as np
from matplotlib import pyplot as plt

n=100
A = np.diag(np.mgrid[:n])
A[:, -1] = A[-1, :] = np.mgrid[:n]

#x = np.flip(x)
B = A[::-1] #reserve array

plt.spy(A)
plt.show()
plt.spy(A[::-1, :])
plt.show()
plt.spy(np.dot(A,A))
plt.show()
plt.spy(np.dot(A,B))

plt.show()
