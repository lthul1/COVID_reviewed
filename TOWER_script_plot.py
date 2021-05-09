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

# n = number of samples paths to simulation
n = 200

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
nc = 25
T = 25
xi = 0.8
lz = 0.1
a = 0.9
b = 0.1
cc = 0.7
dd = 0.3
fn = 0
fp = 0.04
p_inf0 = 0.15
p_rec0 = 0.01
gamma_ = 0.3
alpha_ = 0.8
N = gen_N(nc)
bw = 0.2
locs = gen_locs(nc)
bw_approx = 0.2
FLOW = np.exp(-0.5 * pdist(locs / bw, 'sqeuclidean'))
FLOW = squareform(FLOW)
FLOW = np.min(
    [0.8 * np.ones(FLOW.shape), np.max([0.001 * np.ones(FLOW.shape), FLOW], axis=0)],
    axis=0)
np.fill_diagonal(FLOW, 0)

hyperparameters = [nc, T, xi, lz, a, b, cc, dd, fn, fp, p_inf0, p_rec0, gamma_, alpha_, N, bw, locs, bw_approx, FLOW]

# initialize process classes
vac_proc = vaccine_process2(nc, T, N)
test_proc = test_process(nc, T)
# create alias for the methods
vac_fun = vac_proc.stoch
test_fun = test_proc.const

# initial controller state


# list of vaccine policies
vaccine_policies = [vac_policies.prop_policy, vac_policies.Sampled_greedy,  vac_policies.risk_DLA_prime, vac_policies.susc_allocate, vac_policies.projectionDLA]
mv = len(vaccine_policies)
vac_names = ['even', 'sampledCFA', 'DLA-2', 'PFA', 'Qproj']
# vparams0 is null policy params
vparams0 = []
vparams1 = [50, 10]
vparams2 = []
vparams3 = [0.05]
vparams4 = [0.85, 0.4]
vparams5 = [6, 2500, 0.5]

vparam_list = [vparams0, vparams0, vparams3, vparams1, vparams3]

# list of testing policies
testing_policies = [test_policies.EI, test_policies.REMBO_EI, test_policies.prop_greedy_trade]
mt = len(testing_policies)
test_names = ['EI', 'REMBO', 'tradeoff']
# tparams0 is null policy params
tparams0 = []
tparams1 = [10]
tparams2 = [3]
tparams3 = [0.3]
tparam_list = [tparams0, tparams2, tparams3]

betahat = gen_betas(nc, T, n)
rs =  np.ones(n)
COlist = [[[] for j in range(mv)] for i in range(mt)]
Xvaclist = [[[] for j in range(mv)] for i in range(mt)]
Inc = [[[] for j in range(mv)] for i in range(mt)]
for i in range(mt):
    for j in range(mv):
        for k in range(n):
            # stochastic list
            # stochastics[0] = betahat[k] - beta values for sample k
            # stochastics[1] = vac_fun - vaccine production function
            # stochastics[2] = test_fun - test kit production function
            print('k = '+str(k) + ':   Vaccine Policy: ' + str(vac_names[j]))
            Movers = gen_FLOW(nc, T, n, FLOW, N)
            stochastics = [betahat[:, :, k], vac_fun, test_fun, Movers, rs[k]]
            costs, xvaclist, Ilist = run_sample_path(hyperparameters, stochastics, vaccine_policies[j], testing_policies[i],
                                    vparam_list[j], tparam_list[i])
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
      'honeydew', 'plum', 'darkturquoise', 'navy', 'slategrey', 'aquamarine','m', 'c', 'y','m', 'c', 'y',]


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

vac_co = ['r', 'b', 'g', 'm',  'lightsteelblue', 'purple', 'teal', 'olive', 'pink',
      'honeydew', 'plum']

costs = COlist.copy()
HH = np.arange(len(test_names))
fig = plt.figure()
fig.set_size_inches(5.5, 3.5)
ax = fig.add_axes([0,0,1,1])
legend = []
counter = 0
Ntot = np.sum(N) * np.ones(T)
S = []
Ser = []
for k in range(len(vac_names)):
    ts_res = []
    ts_err = []
    for j in range(len(test_names)):
        cumulative_eval = np.cumsum(costs[j][k], axis=1)

        summation_eval_samp = np.cumsum(cumulative_eval, axis=1)
        summation_eval_mean = np.mean(summation_eval_samp, axis=0)
        summation_eval_std = np.std(summation_eval_samp, axis=0)
        # ax[0].plot(instant_mean, co[counter])
        ts_res.append(summation_eval_mean[T-1])
        ts_err.append(summation_eval_std[T - 1])
        legend.append([vac_names[k] + ' + ' + test_names[j]])
        counter += 1
    S.append(ts_res)
    Ser.append(ts_err)
# ax[0].legend(legend)

for k in range(len(vac_names)):
    ax.bar(HH + k * 0.1, S[k] - 0.9*np.min(S), yerr=Ser[k], color = vac_co[k], width = 0.1)

ax.legend(labels=vac_names)
ax.set_xticklabels(test_names)

Dvec = []
Dvec.append(S)
Dvec.append(Ser)
Dvec.append(Inc)
Dvec.append(Xvaclist)
now = datetime.now()

current_time = now.strftime("%H,%M,%S")
dl.save_data(parameter_list, 'data/paramlist_'+str(current_time)+'.obj')
dl.save_data(Dvec, 'data/dataset_'+str(current_time)+'.obj')





plt.show()
