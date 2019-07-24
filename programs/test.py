# coding=utf-8
import numpy as np
lst = [10, 5, 20]
x = (10 + sum((i*2 + i+10) for i in lst))
print(x)