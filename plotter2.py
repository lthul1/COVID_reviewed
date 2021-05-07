import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt
import datetime


time = '06,11,02'
null_time = '06,39,16'
null_costs = dl.load_data('data/dataset_'+str(null_time)+'.obj')
costs = dl.load_data('data/dataset_'+str(time)+'.obj')
parameter_list = dl.load_data('data/paramlist_'+str(time)+'.obj')


vac_names = parameter_list[21]
test_names = parameter_list[22]
N = parameter_list[14]
T = parameter_list[1]

S = costs[0]
Serr = costs[1]
co = ['k', 'r', 'b', 'g', 'm', 'c', 'y', 'grey', 'indigo', 'lightsteelblue', 'purple', 'teal', 'olive', 'pink',
      'honeydew', 'plum', 'darkturquoise', 'navy', 'slategrey', 'aquamarine','m', 'c', 'y','m', 'c', 'y',]


HH = np.arange(len(test_names))
fig = plt.figure()
fig.set_size_inches(5.5, 3.5)
ax = fig.add_axes([0,0,1,1])
legend = []
counter = 0
Ntot = np.sum(N) * np.ones(T)

# DES = (null_costs[0][0] - S) / null_costs[0][0]


for k in range(len(vac_names)):
    ck = (null_costs[0][0][0] - S[k]) / null_costs[0][0][0]
    print('ck = '+str(ck))
    ax.bar(HH + k * 0.1, ck, yerr = Serr[k]/null_costs[0][0][0], color = co[k], width = 0.1)

ax.legend(labels=vac_names)
ax.set_xticklabels(test_names)




plt.show()

