import numpy as np
import matplotlib.pyplot as plt
import Simulator
import Controller
import vac_policies
import test_policies
from path_function import *
from utility import *
import data_loader as dl
import time
from datetime import datetime
import cost_object
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

# n = number of samples paths to simulation
n = 5
df = 1
# list of hyperparameters
# hyperparameters[0] - nc = number of zones
# hyperparameters[1] - T = time horizon
# hyperparameters[2] - xi = vaccine efficacy
# hyperparameters[3] - lz = beta-binomial update weight
# hyperparameters[4] - a = probability of having symptoms given positive
# hyperparameters[5] - b = probability of having symptoms given negative
# hyperparameters[6] - cc = base probability of being tested given symptomatic
# hyperparameters[7] - dd = base probability of being tested given asymptomatic
# hyperparameters[8] - fn = probability of false negative test
# hyperparameters[9] - fp = probability of false positive test
# hyperparameters[10] - p_inf0 - initial infected population
# hyperparameters[11] - p_rec0 - initial immune population
# hyperparameters[12] - gamma_ - mean recovery rate
# hyperparameters[13] - alpha_ - mobility constants
# hyperparameters[14] - N = population vector
# hyperparameters[15] - bw = mobility bandwidth
# hyperparameters[16] - locs = locations of each zone
# hyperparameters[17] - bw_approx - initial controller approximated bandwidth
nc = 51
T = 22
xi = 0.9
lz = 0.1
a = 0.9
b = 0.1
cc = 0.7
dd = 0.3
fn = 0
fp = 0.04
ir = np.zeros(nc)
ii = np.random.randint(low=0,high=nc,size=3)
ir[ii] = 1
p_inf0 = 0.15 * ir
p_rec0 = 0.05
gamma_ = 0.15
alpha_ = 0.8
N = dl.load_data('USA_DATA/state_population.obj')
N = np.int32(np.array(N))
N = N.astype(float)
bw = 0.2
locs = gen_locs(nc)
bw_approx = 0.2
FLOW = dl.load_data('USA_DATA/FLOW_STATE.obj')

hyperparameters = [nc, T, xi, lz, a, b, cc, dd, fn, fp, p_inf0, p_rec0, gamma_, alpha_, N, bw, locs, bw_approx, FLOW]

# initialize process classes
vac_proc = vaccine_process2(nc, T, N)
test_proc = test_process(nc, T, N)
# create alias for the methods
vac_fun = vac_proc.data_vacs
test_fun = test_proc.data_tests

# initial controller state


# list of vaccine policies
vaccine_policies = [vac_policies.prop_policy, vac_policies.Sampled_greedy, vac_policies.risk_DLA_prime, vac_policies.risk_DLA_param]
mv = len(vaccine_policies)
vac_names = ['even', 'CFA', 'DLA-2', 'params']
# vparams0 is null policy params
vparams0 = []
vparams1 = [50, 10]
vparams2 = []
vparams3 = [0.05]
vparams4 = [0.15, 0.3]
vparams5 = [6, 1000, 0.5]
vparams6 = [.05, 4, 0, 2.75, .7]

vparam_list = [vparams0, vparams0, vparams3, vparams6]


# list of testing policies
testing_policies = [test_policies.null_policy, test_policies.EI, test_policies.prop_greedy_trade]
mt = len(testing_policies)
test_names = ['null', 'EI', 'Fairness']
# tparams0 is null policy params
tparams0 = []
tparams1 = [10]
tparams2 = [3]
tparams3 = [0.3]
tparam_list = [tparams0, tparams0, tparams3]

betahat = gen_USA_betas(nc, T, n)



Movers = [gen_FLOW(nc, T, n, FLOW, N) for i in range(n)]
COlist = [[[] for j in range(mv)] for i in range(mt)]
Xvaclist = [[[] for j in range(mv)] for i in range(mt)]
Inc = [[[] for j in range(mv)] for i in range(mt)]
# seeds = np.random.randint(low=0, high=500, size=n)
seeds = np.random.randint(low=0, high=500, size=n)
runtimes = [[[] for j in range(mv)] for i in range(mt)]
for i in range(mt):
    for j in range(mv):
        for k in range(n):
            # stochastic list
            # stochastics[0] = betahat[k] - beta values for sample k
            # stochastics[1] = vac_fun - vaccine production function
            # stochastics[2] = test_fun - test kit production function
            print('k = '+str(k))
            start = cdate(datetime.now())
            rs = 1
            Movers = gen_FLOW(nc, T, n, FLOW, N)
            stochastics = [betahat[:, :, k], vac_fun, test_fun, Movers, rs, seeds[k]]
            costs, xvaclist, Ilist = run_sample_path(hyperparameters, stochastics, vaccine_policies[j], testing_policies[i],
                                    vparam_list[j], tparam_list[i])
            gg = cdate(datetime.now())
            runtimes[i][j].append(gg - start)
            print('Vaccine Policy: '+str(vac_names[j]) + '  , Test Policy: ' +str(test_names[i]) + '  time:  ' +str(gg - start))
            COlist[i][j].append(costs.inst_list)
            Xvaclist[i][j].append(xvaclist)
            Inc[i][j].append(Ilist)

# plotter = cost_object.plotter(hyperparameters, vac_names, test_names)
# plotter.plot_all(COlist)
# plt.show()

parameter_list = hyperparameters.copy()
parameter_list.append(vparam_list)
parameter_list.append(tparam_list)
parameter_list.append(vac_names)
parameter_list.append(test_names)

vac_names = parameter_list[21]
test_names = parameter_list[22]
N = parameter_list[14]
T = parameter_list[1]
co = ['k', 'r', 'b', 'g', 'm', 'c', 'y', 'grey', 'indigo', 'lightsteelblue', 'purple', 'teal', 'olive', 'pink',
      'honeydew', 'plum', 'darkturquoise', 'navy', 'slategrey', 'aquamarine', 'm', 'c', 'y','m', 'c', 'y', 'lightsteelblue', 'purple', 'teal', 'olive', 'pink']


costs = COlist.copy()
fig2, ax2 = plt.subplots(1, 2)
fig2.set_size_inches(14.5, 8.5)
legend2 = []
counter2 = 0
Ntot = np.sum(N) * np.ones(T)
for j in range(len(test_names)):
    for k in range(len(vac_names)):
        cumulative = np.cumsum(costs[j][k], axis=1)
        instant_mean = np.mean(costs[j][k], axis=0)
        cumulative_mean = np.mean(cumulative, axis=0)
        summation_mean = np.cumsum(cumulative_mean)
        # ax[0].plot(instant_mean, co[counter])
        ax2[0].plot(cumulative_mean, co[counter2])
        ax2[1].plot(summation_mean, co[counter2])
        legend2.append([vac_names[k] + ' + ' + test_names[j]])
        counter2 += 1
ax2[0].legend(legend2)

vac_co = ['r', 'b', 'g', 'm',  'lightsteelblue', 'purple', 'teal', 'olive', 'pink', 'honeydew', 'plum']

costs = COlist.copy()
HH = np.arange(len(test_names))
fig = plt.figure()
fig.set_size_inches(5.5, 3.5)
ax = fig.add_axes([0,0,1,1])
legend = []
counter = 0
Ntot = np.sum(N) * np.ones(T)
S = []
S2 = []
Ser = []
S2er = []
Mruntime = [[[] for j in range(mv)] for i in range(mt)]
Vruntime = [[[] for j in range(mv)] for i in range(mt)]
for k in range(len(vac_names)):
    ts_res = []
    ts2_res = []
    ts_err = []
    ts2_err = []
    for j in range(len(test_names)):
        cumulative_eval = np.cumsum(costs[j][k], axis=1)
        summation_eval_samp = np.cumsum(cumulative_eval, axis=1)
        summation_eval_mean = np.mean(summation_eval_samp, axis=0)
        summation_eval_std = np.std(summation_eval_samp, axis=0)

        cummax_samps = np.max(cumulative_eval, axis=1)
        cummax_mean = np.mean(cummax_samps, axis=0)
        cummax_std = np.std(cummax_samps, axis=0)
        print(vac_names[k] + ' + ' + test_names[j] + ' = ' + str(np.mean(runtimes[j][k])) + ' +- ' + str(np.std(runtimes[j][k])))
        Mruntime[j][k] = np.mean(runtimes[j][k])
        Vruntime[j][k] = np.std(runtimes[j][k])

        # ax[0].plot(instant_mean, co[counter])
        ts_res.append(summation_eval_mean[T-1])
        ts_err.append(summation_eval_std[T - 1])
        ts2_res.append(cummax_mean)
        ts2_err.append(cummax_std)
        legend.append([vac_names[k] + ' + ' + test_names[j]])
        counter += 1
    S.append(ts_res)
    Ser.append(ts_err)
    S2.append(ts2_res)
    S2er.append(ts2_err)
# ax[0].legend(legend)

for k in range(len(vac_names)):
    ax.bar(HH + k * 0.1, S[k] - 0.8*np.min(S), yerr=Ser[k], color = vac_co[k], width = 0.1)

ax.legend(labels=vac_names)
ax.set_xticklabels(test_names)
ax.set_title('Summation')

# fig2 = plt.figure()
# fig2.set_size_inches(5.5, 3.5)
# ax2 = fig2.add_axes([0,0,1,1])
#
# for k in range(len(vac_names)):
#     ax2.bar(HH + k * 0.1, S2[k], yerr=S2er[k], color = vac_co[k], width = 0.1)
#
# ax2.legend(labels=vac_names)
# ax2.set_xticklabels(test_names)
# ax.set_title('Max')



Dvec = []
Dvec.append(S)
Dvec.append(Ser)
Dvec.append(Inc)
Dvec.append(Xvaclist)
Dvec.append(Mruntime)
Dvec.append(Vruntime)
now = datetime.now()

current_time = now.strftime("%H,%M,%S")
dl.save_data(parameter_list, 'USA_OUTPUT/paramlist_'+str(current_time)+'.obj')
dl.save_data(Dvec, 'USA_OUTPUT/dataset_'+str(current_time)+'.obj')

plt.show()
