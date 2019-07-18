import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

x = [None] * 10
y = [None] * 10
z = [None] * 10

for i in range(len(x)):
    x[i] = i
    y[i] = np.exp(-0.2*i)*np.cos(i) 
    z[i] = np.exp(-0.2*i)*np.cos(i) 


ax.plot(x, y, z, label='curve')
ax.legend()

plt.show()
