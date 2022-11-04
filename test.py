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

print(hash((0,7,4)))
# x = [[random.random() for i in range(N)] for j in range(N)]
# print(hash(x))
# print(hash(('aab',2,2)))

x = np.array([[1,2,3], [1,12,3]])
x[x > 2 and x < 10] = 7
print(x)