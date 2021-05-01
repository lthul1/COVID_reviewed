import numpy as np
import matplotlib.pyplot as plt
import data_loader as dl



pdata = dl.load_data('tuning_data/params_riskCFA_14,19,02.obj')
costs = dl.load_data('tuning_data/costs_riskCFA_14,19,02.obj')


fig, ax = plt.subplots(1,2)
pl = np.array(pdata)
ax[0].plot(pl[:,0])
# ax[0].plot(pl[:,1])
# ax[0].plot(pl[:,2])
ax[1].plot(costs[1:])



plt.show()


