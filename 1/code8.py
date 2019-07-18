import numpy as np
import matplotlib.pyplot as plt


x = [None] * 10
y = [None] * 10

for i in range(len(x)):
    x[i] = i
    y[i] = np.exp(-0.2*i)*np.cos(i) 


fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.axis([-1,1,-1,1])
ax1.plot(x,y,"b")
 
ax2.plot(x,y,"b")
ax2.axis([0,2.3,4,5])

ax3.axis([-1,1,0,5])
ax3.loglog(x, y)
ax3.plot(x,y,"b")

plt.show()
