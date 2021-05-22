import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt
from utility2 import *

time = '18,11,54'
# time = '05,33,41'
null_time = '14,39,04'


costs = dl.load_data('USA_OUTPUT/dataset_'+ str(time)+'.obj')
parameter_list = dl.load_data('USA_OUTPUT/paramlist_'+ str(time)+'.obj')

ncosts = dl.load_data('USA_OUTPUT/dataset_'+ str(null_time)+'.obj')
nparameter_list = dl.load_data('USA_OUTPUT/paramlist_'+ str(null_time)+'.obj')

vac_names = parameter_list[21]
test_names = parameter_list[22]
N = parameter_list[14]
T = parameter_list[1]
co = ['k', 'r', 'b', 'g', 'm', 'c', 'y', 'grey', 'indigo', 'lightsteelblue', 'purple', 'teal', 'olive', 'pink', 'honeydew', 'plum', 'darkturquoise', 'navy', 'slategrey', 'aquamarine']


vac_co = ['r', 'b', 'g', 'm',  'lightsteelblue', 'purple', 'teal', 'olive', 'pink', 'honeydew', 'plum']

HH = np.arange(len(test_names))
fig, ax = plt.subplots(1,1)

legend = []
counter = 0
Ntot = np.sum(N) * np.ones(T)

S = costs[0]
Ser = costs[1]
Inc = costs[2]
X = costs[3]
mr = costs[4]
mv = costs[5]

nullS = ncosts[0][0]
nullSer = ncosts[1][0]

legend = []
for k in range(len(vac_names)):
    nullSt = np.array(nullS)
    St = np.array(S[k])
    Sert = np.array(Ser[k])
    mm = (nullSt - St) / nullSt
    ss = Sert/nullSt
    ax.bar(HH + k * 0.15, list(mm), yerr=list(ss), color = vac_co[k][:3], width = 0.15)
    for t in range(len(HH)):
        print(vac_names[k] + ' + ' + test_names[t] + ' = ' + str(mr[t][k]) + ' +- ' + str(mv[t][k]))
        # print(vac_names[k] + ' + ' + test_names[t] + ' = ' + str(S[k][t]) +' +- ' + str(Ser[k][t]))
ax.set_title('Policy Performance for USA Scenario')
# ax[1].set_ylim([0.85*np.min(S), np.max(S)])
ax.xaxis.set_ticks_position('none')
ax.set_ylim([0.4, 0.6])
ax.legend(['PFA', 'CFA', 'DLA-2', 'DLA-PARAM'])

scheme = 'diff'
Differences = np.mean(Inc[0][0], axis=0) - np.mean(Inc[0][3], axis=0)


plt.figure()
# plt.plot(np.cumsum(np.sum(Differences, axis=1)))
USA_plot(Differences, T=T, scheme=scheme)
# plt.figure()

# plt.plot(np.mean(X[0][3], axis=0))
# print(np.sum(np.mean(X[0][0], axis=0), axis=1))



plt.show()

