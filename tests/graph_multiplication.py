import numpy as np
import pandas as pd

a = [1, 2, 3]
b = [4, 5, 6]
c = []


for i in range(len(a)):
    c.append(a[i] * b[i])
print(c)
