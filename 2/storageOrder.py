import numpy as np

A = np.array([[1,2], [3,4]])
B = np.array([[1,2], [3,4]], order='F')

np.ravel(A, 'K')
np.ravel(B, 'K')

np.ravel(A.T, 'K')
np.ravel(B.T, 'K')

A.flags['C_CONTIGUOUS']
B.flags['F_CONTIGUOUS']
A.T.flags['F_CONTIGUOUS']
B.T.flags['C_CONTIGUOUS']