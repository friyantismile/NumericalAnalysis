import numpy as np

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