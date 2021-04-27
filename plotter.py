import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt

costs = dl.load_data('data/COdata_1_3_1619528900.406901_.obj')
parameter_list = dl.load_data('data/params_1_3_1619528900.409087_.obj')

vac_names = parameter_list[20]
test_names = parameter_list[21]
N = parameter_list[14]
T = parameter_list[1]
co = ['k', 'r', 'b', 'g', 'm', 'c', 'y', 'grey', 'indigo', 'lightsteelblue', 'purple', 'teal', 'olive', 'pink', 'honeydew', 'plum', 'darkturquoise', 'navy', 'slategrey', 'aquamarine']


fig, ax = plt.subplots(1,2)
fig.set_size_inches(18.5, 10.5)
legend = []
counter = 0
Ntot = np.sum(N) * np.ones(T)
for j in range(len(test_names)):
    for k in range(len(vac_names)):
        cumulative = np.cumsum(costs[j][k], axis=1)
        instant_mean = np.mean(costs[j][k], axis=0)
        cumulative_mean = np.mean(cumulative,axis=0)
        summation_mean = np.cumsum(cumulative_mean)
        # ax[0].plot(instant_mean, co[counter])
        ax[0].plot(cumulative_mean, co[counter])
        ax[1].plot(summation_mean, co[counter])
        legend.append([vac_names[k] + ' + ' + test_names[j]])
        counter+=1
ax[0].legend(legend)

plt.show()

