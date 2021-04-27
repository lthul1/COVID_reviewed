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
nc = 18
T = 20
xi = 1
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

vaccine_policy = vac_policies.risk_greedy
vac_tuner = 'riskCFA'
test_policy = test_policies.null_policy
thetaDLA = 0.7
tunable0 = np.array([lz, thetaDLA])

K = 50
hh = 1
eta = np.arange(1,K+1)
eta = np.max([0.01 * np.ones(eta.shape), hh/(eta+hh)], axis=0)

tunable1 = tunable0.copy()
grad_squared = 0
lr = 0.8
tlist = [list(tunable1)]
Fl = [0]
print('tunable1 = ' + str(tunable1))
for k in range(K):
	print('###  k = '+str(k) + '  ####')
	np.random.seed(np.random.randint(100))
	betahat = gen_betas(nc, T, n)
	Movers = gen_FLOW(nc, T, n, FLOW, N)
	stochastics = [betahat[:, :, 0], vac_fun, test_fun, Movers]
	tune_t = tunable1.copy()
	vk = np.random.normal(size=tunable0.shape)
	tune_t1 = tune_t + eta[k] * vk
	boolt1 = tune_t1 <= 0
	boolt2 = tune_t1 >= 1
	tune_t1[boolt1] = 0.001
	tune_t1[boolt2] = 1
	hyperparameters_t = hyperparameters.copy()
	hyperparameters_t1 = hyperparameters.copy()
	hyperparameters_t[3] = tune_t[0]
	hyperparameters_t1[3] = tune_t1[0]
	tparams_t = []
	tparams_t1 = []
	vparams_t = [tune_t[1]]
	vparams_t1 = [tune_t1[1]]
	COt = run_sample_path(hyperparameters_t, stochastics, vaccine_policy, test_policy, vparams_t, tparams_t)
	COt1 = run_sample_path(hyperparameters_t1, stochastics, vaccine_policy, test_policy, vparams_t1, tparams_t1)

	ct = COt.inst_list
	ct1 = COt1.inst_list

	cumulativet = np.cumsum(ct)
	summation_meant = np.cumsum(cumulativet)

	cumulativet1 = np.cumsum(ct1)
	summation_meant1 = np.cumsum(cumulativet1)

	Ft = summation_meant[T-1]
	Ft1 = summation_meant1[T-1]

	G = (Ft1 - Ft)/eta[k] * vk
	print('G = ' + str(G))
	print('Ft = ' + str(Ft))
	grad_squared = 0.9 * grad_squared + 0.1 * G * G
	alpha = np.max([1e-6 * np.ones(grad_squared.shape), lr/grad_squared], axis=0)
	tunable1 = tune_t - alpha * G

	#project back into parameter space
	bool1 = tunable1 <= 0
	bool2 = tunable1 >= 1
	tunable1[bool1] = 0.001
	tunable1[bool2] = 1
	print('tunable1 = ' + str(tunable1))
	print('alpha = ' + str(alpha))
	print('eta = ' + str(eta[k]))
	tlist.append(list(tunable1))

	# samples = 5
	# betahat = gen_betas(nc, T, samples)
	# clist = []
	# for i in range(samples):
	# 	stochastics = [betahat[:, :, i], vac_fun, test_fun]
	# 	tune_eval = tunable1.copy()
	# 	hyperparameters_eval = hyperparameters.copy()
	# 	hyperparameters_eval[3] = tune_eval[0]
	# 	tparams_eval = []
	# 	vparams_eval = [tune_eval[1]]
	# 	COeval = run_sample_path(hyperparameters_eval, stochastics, vaccine_policy, test_policy, vparams_eval, tparams_eval)
	# 	clist.append(COeval.inst_list)
	# cumulative_eval = np.cumsum(clist, axis=1)
	# cumulative_mean_eval = np.mean(cumulative_eval, axis=0)
	# summation_eval = np.cumsum(cumulative_mean_eval)
	Fl.append(Ft)

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
dl.save_data(tlist, 'tuning_data/params_'+str(vac_tuner)+'_'+str(current_time)+'_.obj')
dl.save_data(Fl, 'tuning_data/costs_'+str(vac_tuner)+'_'+str(current_time)+'_.obj')







