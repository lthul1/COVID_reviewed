import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from utility import gradproj

from utility import proj


x = np.random.rand(10)
g = 20

one = proj(x/g,g)

two = gradproj(x,g)
print(str(x))
print(str(one*g))
print(str(two))
print(str(np.sum(one*g)))

print(str(np.sum(two)))












