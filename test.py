import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
 
plt.style.use('fivethirtyeight')
 
x_val = []
y_val = []
 
index = count()
 
def animate(i):
    x_val.append(next(index))
    y_val.append(random.randint(0,5))
    plt.cla()
    plt.plot(x_val, y_val)
 
# ani = FuncAnimation(plt.gcf(), animate)
 


for _ in range(10):
    print('aaa')
    plt.pause(0.001)
    animate(0)
    
print('bbb')
plt.tight_layout()
plt.show()    