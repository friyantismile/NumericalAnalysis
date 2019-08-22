import numpy as np
import math
import matplotlib.pyplot as plt
  
x = [None] * 10
y = [None] * 10

for i in range(len(x)):
    x[i] = i
    y[i] = math.exp(-0.2*i)*math.cos(i) 

plt.subplot(311)
plt.plot(x,y,"+-g")
plt.xlabel('Linear x axis')


plt.subplot(312)
plt.loglog(x,y,"+-g")
plt.xscale('symlog')
plt.yscale('symlog')


plt.legend()
plt.tight_layout()
plt.show()