import numpy as np
import math
import matplotlib.pyplot as plt
  
x = [None] * 10
y = [None] * 10

for i in range(len(x)):
    x[i] = i
    y[i] = math.exp(-0.2*i)*math.cos(i) 


# x = [20, 200, 2000, 20000, 200000]
# y = [30, 300, 3000, 30000, 300000]

# fig, (ax, ax1, ax2) = plt.subplots(3)
# plt.yscale('log')

# ax.loglog(x,y)
# #ax.loglog(x[0],y[0], '--')

# #ax.loglog([x[0],y[-1]],[x[-1],y[0]], '--')

# ax1.semilogy(x,y)
# #ax1.semilogy(x[1],y[1], "--")

# ax2.semilogx(x,y)
# #ax2.semilogx(x[2],y[2], "--")


# plt.show()
# dt = 0.01
# x = np.arange(-50.0, 50.0, dt)
# y = np.arange(0, 100.0, dt)


plt.subplot(311)
plt.semilogx(x, y)
plt.xscale('symlog')
plt.ylabel('symlogx')
plt.grid(True)
plt.gca().xaxis.grid(True, which='minor')  # minor grid on too

plt.subplot(312)
plt.semilogy(y, x)
plt.yscale('symlog')
plt.ylabel('symlogy')

plt.subplot(313)
plt.loglog(x, y)
plt.xscale('symlog')
plt.yscale('symlog')
plt.grid(True)
plt.ylabel('loglog')
 
plt.tight_layout()
plt.show()