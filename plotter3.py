import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt
import datetime
from utility import *


time = '14,27,05'
params = dl.load_data('USA_output/paramlist_'+str(time)+'.obj')
data = dl.load_data('USA_output/dataset_'+str(time)+'.obj')

T = 25

USA_plot(data, T)





plt.show()

