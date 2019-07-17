import numpy as np
import math
import matplotlib.pyplot as plt

def drawGrid(isGrid, plt, x, y, gridType="-", gridCol="r", **kwargs):
    fig, ax = plt.subplots()
    plt.plot(x,y)
    plt.title("Code 2")
    
    ax.grid(b=isGrid,linestyle=gridType,color=gridCol, **kwargs)
    
x = [None] * 10
y = [None] * 10

for i in range(len(x)):
    x[i] = i
    y[i] = math.exp(-0.2*i)*math.cos(i) 

drawGrid(None, plt, x, y,"dashed", "r")

plt.show()




