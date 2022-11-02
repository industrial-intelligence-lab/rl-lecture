data=[]
data.append((1, 10, 20))
data.append((2, 11, 21))
data.append((3, 12, 22))
data.append((4, 13, 23))
data.append((5, 14, 24))

# print( (i for i, t in enumerate(data) if t[0] == 4) )
# print( next(i for i, t in enumerate(data) if t[0] == 1) )


for i, (a,b,c) in enumerate(data):
    print(i, '->', b)

x = (1,2,3)
a,b,c = x
print(a,b,c)

# first_occurence_idx = next(x for i, x in enumerate(data))
# print(first_occurence_idx)

# # tuple 선언
# x = (1,2,3)
# # unpacking
# a, b, c = x
# print(b)


a = 100
b = 1 if a > 0 and a > 10 else -1 
print(b)

x = [1,2,3,4,5,6,7,9,10]
y = [i for i in x if i>3]
print(y)

import numpy as np
from collections import defaultdict
# N = defaultdict(lambda: np.zeros(10))
# print(N['b'])
# print(N['b'][1])
# N['a'][0] = 100
# print(N['a'])

PI = defaultdict(lambda: np.ones(4)/4)      # [state][action] 
print(PI['a'])
print(np.argmax(PI['a']))

x = np.ones(4)
print(x / 4)


from Person import Human 

p1 = Human('park', 32)
print(p1.name)

location = [int(n, 10) for n in "1,2,3,4,5".split(",")]
print(location)