import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


x = [None] * 10


for i in range(len(x)):
    x[i] = i+1

fx1 = [None] * 10
for i in range(len(fx1)):
    fx1[i] = (3 * np.power(x[i],2) - 4.5 / x[i]) * np.exp(-x[i] / 1.3)

fx2 = [None] * 10
for i in range(len(fx2)):
    fx2[i] = 5*np.sin(5*x[i])*np.exp(-x[i])

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(x,fx1,"r")
ax1.plot(x,fx2,"b", "5*np.sin(5*x[i])*np.exp(-x[i])")
ax1.axis([0.5,5,-5,5])
# ax1.set_xlim([0.5,5])
# ax1.set_ylim([-5,5])

ax2.plot(x,x,"b", "Benchmark")
ax2.plot(x,np.power(x,2),"k--", "O(x^2)")

plt.show()
