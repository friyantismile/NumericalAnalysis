import numpy as np
from matplotlib import pyplot as plt

def col_wise(A):
    for j in range(A.shape[1] - 1):
        A[:, j + 1] -= A[:, j]

plt.spy(A)
plt.show()
plt.spy(A[::-1, :])
plt.show()
plt.spy(np.dot(A,A))
plt.show()
plt.spy(np.dot(A,B))

plt.show()
