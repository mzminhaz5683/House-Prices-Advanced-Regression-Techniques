# coding=utf-8
import numpy as np
lst = np.array([10, 5, 20])
a = 1/7
x = np.floor(a*lst)
#x = np.floor(sum((a*i) for i in lst))
print(x)