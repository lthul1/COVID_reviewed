import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt
import datetime
from utility import *


time = '10,52,21'
params = dl.load_data('USA_output/paramlist_'+str(time)+'.obj')
data = dl.load_data('USA_output/dataset_'+str(time)+'.obj')

T = 25
i = 0
j = 1
S = data[0]
Serr = data[1]
C = data[2]
I = data[3]
costs = I[i][j]


Ibar = np.mean(np.array(I[i][j]), axis=0).T

print(Ibar)


USA_plot(Ibar, T)





plt.show()

