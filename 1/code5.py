import numpy as np
import math
import matplotlib.pyplot as plt

def vectorSize(vector):
    return len(vector)

def vectorValue(x,y):
    return len(vector)
  
x = [None] * 10
y = [None] * 10

for i in range(len(x)):
    x[i] = i
    y[i] = math.exp(-0.2*i)*math.cos(i) 

plt.subplot(311)
plt.plot(x,y,"g--")

plt.subplot(312)
plt.plot(x,y)

plt.subplot(313)
plt.plot(x,y, ".r", "data w/ red dots")

#plt.axis([0, 6, 0, 20])

plt.tight_layout()
plt.show()

