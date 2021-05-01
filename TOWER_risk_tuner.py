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
nc = 25
T = 25
xi = 0.8
lz = 0.4
a = 0.9
b = 0.1
cc = 0.7
dd = 0.3
fn = 0
fp = 0
p_inf0 = 0.1
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

hyperparameters = [nc,T,xi,lz,a,b,cc,dd,fn,fp,p_inf0,p_rec0,gamma_,alpha_,N,bw,locs,bw_approx,FLOW]

# initialize process classes
vac_proc = vaccine_process(nc, T)
test_proc = test_process(nc, T)
# create alias for the methods
vac_fun = vac_proc.stoch
test_fun = test_proc.const

vaccine_policy = vac_policies.risk_DLA_prime
vac_tuner = 'nonlinear'
test_policy = test_policies.REMBO_EI

param0 = np.arange(0.05, 0.95, .05)
param1 = np.arange(2,3,1)


parameterlist = np.vstack(np.meshgrid(param0,param1)).reshape(2,len(param0)*len(param1)).T
Fl = []
Fl_mean = []
Fl_std = []
samples = 6
cc = np.random.randint(100, size= samples)
for k in range(parameterlist.shape[0]):
	print('###  k = '+str(k) + ' of '+str(parameterlist.shape[0]) +'  ####')
	betahat = gen_betas(nc, T, samples)
	clist = []
	for i in range(samples):
		np.random.seed(cc[i])
		Movers = gen_FLOW(nc, T, n, FLOW, N)
		stochastics = [betahat[:, :, i], vac_fun, test_fun, Movers]
		tune_eval = parameterlist[k,:]
		hyperparameters_eval = hyperparameters.copy()
		hyperparameters_eval[3] = 0.2
		tparams_eval = [tune_eval[1]]
		vparams_eval = [tune_eval[0]]
		COeval = run_sample_path(hyperparameters_eval, stochastics, vaccine_policy, test_policy, vparams_eval, tparams_eval)
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
dl.save_data(parameterlist, 'tuning_data/params_'+str(vac_tuner)+'_'+str(current_time)+'.obj')
dl.save_data(Fl, 'tuning_data/costs_'+str(vac_tuner)+'_'+str(current_time)+'.obj')
dl.save_data(Fl_mean, 'tuning_data/mcosts_'+str(vac_tuner)+'_'+str(current_time)+'.obj')
dl.save_data(Fl_std, 'tuning_data/scosts_'+str(vac_tuner)+'_'+str(current_time)+'.obj')

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
pflag = True
if pflag:
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(M, origin='lower')
    ax[1].imshow(M1, origin='lower')
    ax[2].imshow(M2, origin='lower')
# plt.plot(M.flatten())
else:
    ax = plt.axes(projection='3d')
    X,Y = np.meshgrid(param0, param1)
    ax.plot_surface(X,Y,M1)
fig2,ax2 = plt.subplots(1,1)
ax2.plot(Fl)
plt.show()


