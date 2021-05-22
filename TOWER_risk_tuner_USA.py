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
n = 1

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
xi = 0.8
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
p_rec0 = 0.15
gamma_ = 0.2
alpha_ = 0.8
N = dl.load_data('USA_DATA/state_population.obj')
N = np.int32(np.array(N))
N = N.astype(float)
bw = 0.2
locs = gen_locs(nc)
bw_approx = 0.2
FLOW = dl.load_data('USA_DATA/FLOW_STATE.obj')

hyperparameters = [nc,T,xi,lz,a,b,cc,dd,fn,fp,p_inf0,p_rec0,gamma_,alpha_,N,bw,locs,bw_approx,FLOW]

# initialize process classes
vac_proc = vaccine_process2(nc, T, N)
test_proc = test_process(nc, T, N)
# create alias for the methods
vac_fun = vac_proc.data_vacs
test_fun = test_proc.data_tests

vaccine_policy = vac_policies.risk_DLA_param
vac_tuner = 'USA'
test_policy = test_policies.pure_exploration


param0 = np.arange(3.25, 6, .25)
param1 = np.arange(2.25, 5, .25)

# param0 = np.arange(0, 1, .5)
# param1 = np.arange(0, .4, .2)

parameterlist = np.vstack(np.meshgrid(param0,param1)).reshape(2,len(param0)*len(param1)).T
Fl = []
Fl_mean = []
Fl_std = []
samples = 10

rs = np.ones(samples)
cc = np.random.randint(100, size= samples)
Movers = [gen_FLOW(nc, T, n, FLOW, N) for i in range(samples)]
betahat = gen_USA_betas(nc, T, samples)
seeds = np.random.randint(low=0, high=500, size=samples)

for k in range(parameterlist.shape[0]):
    print('###  k = '+str(k) + ' of '+str(parameterlist.shape[0]) +'  ####')
    clist = []
    for i in range(samples):
        # np.random.seed(cc[i])
        np.random.seed(None)
        stochastics = [betahat[:, :, i], vac_fun, test_fun, Movers[i], rs[i], seeds[i]]
        tune_eval = parameterlist[k,:]
        hyperparameters_eval = hyperparameters.copy()
        hyperparameters_eval[3] = 0.1
        tparams_eval = [4]
        # current = [0.05, 4, 0.5, 2.75, 0.5, 240
        vparams_eval = [.05, tune_eval[0], 0.3, tune_eval[1], 0.5]
        COeval, xvaclist, Ilist  = run_sample_path(hyperparameters_eval, stochastics, vaccine_policy, test_policy, vparams_eval, tparams_eval)
        clist.append(COeval.inst_list)
    cumulative_eval = np.cumsum(clist, axis=1)
    cumulative_mean_eval = np.mean(cumulative_eval, axis=0)
    summation_eval = np.cumsum(cumulative_mean_eval)

    summation_eval_samp = np.cumsum(cumulative_eval, axis=1)
    summation_eval_mean = np.mean(summation_eval_samp, axis=0)
    summation_eval_std = np.std(summation_eval_samp, axis=0)

    Fl.append(summation_eval[T-1])
    Fl_mean.append(summation_eval_mean[T - 1])
    Fl_std.append(summation_eval_std[T - 1])

now = datetime.now()

current_time = now.strftime("%H,%M,%S")
dl.save_data(parameterlist, 'USA_risk/params_'+str(vac_tuner)+'_'+str(current_time)+'.obj')
dl.save_data(Fl, 'USA_risk/costs_'+str(vac_tuner)+'_'+str(current_time)+'.obj')
dl.save_data(Fl_mean, 'USA_risk/mcosts_'+str(vac_tuner)+'_'+str(current_time)+'.obj')
dl.save_data(Fl_std, 'USA_risk/scosts_'+str(vac_tuner)+'_'+str(current_time)+'.obj')

ids0 = {param0[i]:i for i in range(len(param0))}
ids1 = {param1[i]:i for i in range(len(param1))}

M = np.zeros([len(param0), len(param1)])
M1 = np.zeros([len(param0), len(param1)])
M2 = np.zeros([len(param0), len(param1)])
pdata = parameterlist

costs = Fl
mcosts = Fl_mean
scosts = Fl_std
counter = 0
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ci = ids0[pdata[counter,:][0]]
        cj = ids1[pdata[counter, :][1]]

        M[ci,cj] = costs[counter]
        M1[ci, cj] = mcosts[counter]
        M2[ci, cj] = scosts[counter]
        counter+=1
pflag = 0
if pflag == 0:
    fig,ax = plt.subplots(1,1)
    # extent = [np.min(param0), np.max(param0), np.min(param1), np.max(param1)]
    xticks = [str(param1[i]) for i in range(len(param1))]
    yticks = [str(param0[i]) for i in range(len(param0))]
    ax.imshow(M1, origin='lower')
    ax.set_xticks(np.arange(len(param1)))
    ax.set_yticks(np.arange(len(param0)))
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel('param1')
    ax.set_ylabel('param0')
# plt.plot(M.flatten())
elif pflag == 1:
    ax = plt.axes(projection='3d')
    # t0 = np.arange(len(param0))
    # t1 = np.arange(len(param1))
    X,Y = np.meshgrid(param0,param1)
    ax.plot_surface(X,Y,M1.T)
    ax.set_xlabel('param0')
    ax.set_ylabel('param1')
    ax.set_xticks(param0)
    ax.set_yticks(param1)
else:
    plt.plot(param1, Fl)
# fig2,ax2 = plt.subplots(1,1)
# ax2.plot(Fl)
plt.show()


