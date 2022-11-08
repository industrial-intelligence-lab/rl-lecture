import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import numpy as np

x = [[1,2,3],[4,5,6]]
n = np.array(x)
print(np.average(n, axis=0))