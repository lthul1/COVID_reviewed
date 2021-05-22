import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
import gurobipy as gp
from utility import gradproj

from utility import proj
from utility import gen_betas

means = np.array([10, 7, 2])
betaW = np.array([1/2.5])

prior = np.array([12, 7, 1])
beta = np.array([1/1.5])

nx = np.arange(1,100)

sigma_til = np.sqrt(1/beta[0] - 1/(beta[0] + nx * betaW[0]))

c = -np.abs((means[0] - np.max(means[1:]))/ sigma_til)
f = c * stats.norm.cdf(c,0,1) + stats.norm.pdf(c,0,1)

v = sigma_til * f

plt.plot(v/nx)
plt.show()










