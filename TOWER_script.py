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
nc = 3
T = 30
xi = 1
lz = 0.3
a = 0.9
b = 0.1
cc = 0.7
dd = 0.3
fn = 0
fp = 0
p_inf0 = 0.2 * np.random.rand(nc)
p_rec0 = 0.01
gamma_ = 0.3
alpha_ = 0.8
N = gen_N(nc)
bw = 0.2
locs = gen_locs(nc)
bw_approx = 0.2

hyperparameters = [nc,T,xi,lz,a,b,cc,dd,fn,fp,p_inf0,p_rec0,gamma_,alpha_,N,bw,locs,bw_approx]


# initialize process classes
vac_proc = vaccine_process(nc, T)
test_proc = test_process(nc, T)
# create alias for the methods
vac_fun = vac_proc.stoch
test_fun = test_proc.const


# initial controller state


# list of vaccine policies
vaccine_policies = [vac_policies.null_policy, vac_policies.susc_allocate, vac_policies.Sampled_greedy,  vac_policies.risk_adjusted_DLA, vac_policies.fairness_risk_adjusted_DLA]
mv = len(vaccine_policies)
vac_names = ['null', 'PFA', 'Sampled_CFA', 'riskDLA', 'fairnessDLA']
# vparams0 is null policy params
vparams0 = []
vparams1 = [50,10]
vparams2 = []
vparams3 = [0.15]
vparams4 = [0.15, 0.3]

vparam_list = [vparams0, vparams1, vparams2, vparams3, vparams4]

# list of testing policies
testing_policies = [test_policies.pure_exploration, test_policies.EI, test_policies.REMBO_EI, test_policies.prop_greedy_trade]
mt = len(testing_policies)
test_names = ['explore', 'EI', 'REMBO', 'prop_greedy']
#tparams0 is null policy params
tparams0 = []
tparams1 = []
tparams2 = [2]
tparams3 = [0.4]
tparam_list = [tparams0, tparams1, tparams2, tparams3]

betahat = gen_betas(nc,T,n)


COlist = [[[] for j in range(mv)] for i in range(mt)]
for i in range(mt):
	for j in range(mv):
		for k in range(n):
			# stochastic list
			# stochastics[0] = betahat[k] - beta values for sample k
			# stochastics[1] = vac_fun - vaccine production function
			# stochastics[2] = test_fun - test kit production function
			print('Vaccine Policy: '+str(vac_names[j]))
			stochastics = [betahat[:,:,k], vac_fun, test_fun]
			costs = run_sample_path(hyperparameters, stochastics, vaccine_policies[j], testing_policies[i], vparam_list[j], tparam_list[i])
			COlist[i][j].append(costs.inst_list)

# plotter = cost_object.plotter(hyperparameters, vac_names, test_names)
# plotter.plot_all(COlist)
# plt.show()
parameter_list = hyperparameters.copy()
parameter_list.append(vparam_list)
parameter_list.append(tparam_list)
parameter_list.append(vac_names)
parameter_list.append(test_names)
dl.save_data(COlist, 'data/COdata_'+str(n)+'_'+str(nc)+'_'+str(time.time())+'_.obj')
dl.save_data(parameter_list, 'data/params_'+str(n)+'_'+str(nc)+'_'+str(time.time())+'_.obj')
