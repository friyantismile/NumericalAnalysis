import numpy as np
import math
import matplotlib.pyplot as plt

x = [None] * 10
y = [None] * 10

for i in range(len(x)):
    x[i] = i
    y[i] = math.exp(-0.2*i)*math.cos(i)

u = np.linspace(0,9,500)
v = np.cos(u)*np.exp(-0.2*u)
matrix = np.matrix([u,v])

plt.figure()
plt.plot(x,y,"+r", label="Sample data")
plt.plot(u,v,"b", label="Function")

plt.title("Sample figure")
plt.legend()
plt.show()
