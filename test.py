import numpy as np
import random

N = 5

x = ((random.random() for i in range(N)) for j in range(N))
# for i in x:
#     for j in i:
#         print(j)
print(hash(x))
x = ((random.random() for i in range(N)) for j in range(N))

print(hash(x))
x = ((random.random() for i in range(N)) for j in range(N))

print(hash(x))
# x = [[random.random() for i in range(N)] for j in range(N)]
# print(hash(x))
# print(hash(('aab',2,2)))