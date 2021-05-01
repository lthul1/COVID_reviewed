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
print(np.random.rand())
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
T = 20
xi = 1
lz = 0.3
a = 0.9
b = 0.1
cc = 0.7
dd = 0.3
fn = 0.01
fp = 0.02
p_inf0 = 0.1
p_rec0 = 0.01
gamma_ = 0.3
alpha_ = 0.8
N = gen_N(nc)
print(np.random.rand())
bw = 0.2
locs = gen_locs(nc)
bw_approx = 0.2
FLOW = np.exp(-0.5 * pdist(locs / bw, 'sqeuclidean'))
FLOW = squareform(FLOW)
FLOW = np.min(
	[0.8 * np.ones(FLOW.shape), np.max([0.001 * np.ones(FLOW.shape), FLOW], axis=0)],
	axis=0)
np.fill_diagonal(FLOW, 0)
print(np.random.rand())
hyperparameters = [nc,T,xi,lz,a,b,cc,dd,fn,fp,p_inf0,p_rec0,gamma_,alpha_,N,bw,locs,bw_approx,FLOW]

# initialize process classes
vac_proc = vaccine_process(nc, T)
test_proc = test_process(nc, T)
# create alias for the methods
vac_fun = vac_proc.stoch
test_fun = test_proc.const

vaccine_policy = vac_policies.risk_greedy
vac_tuner = 'riskCFA'
test_policy = test_policies.pure_exploration

thetaPFA0 = 0.5
tunable0 = np.array([thetaPFA0])

K = 100
L=2
hh = 1
kk = np.arange(1,K+1)
gamma = 0.101
c = .1
# eta = np.max([0.01 * np.ones(eta.shape), hh/(eta+hh)], axis=0)
ck = c/(kk)**gamma
tunable1 = tunable0.copy()
tune_t = tunable1.copy()
mt = 0
vt = 0
b1 = 0.8
b2 = 0.99
eps= 10e-8
g2=0
lr = 10

et = 0.8
tlist = [list(tunable1)]
Fl = [0]
print('tunable1 = ' + str(tunable1))
for k in range(K):
	print('###  k = '+str(k) + '  ####')
	dG = np.zeros(tunable1.shape[0])
	dF = 0
	vk = np.random.rand(tunable1.shape[0])
	hk = 2 * np.int32(vk > 0.5) - 1

	print('hk = ' +str(hk))
	for m in range(L):
		tune_t = tunable1.copy()
		np.random.seed(np.random.randint(100))
		betahat = gen_betas(nc, T, n)
		Movers = gen_FLOW(nc, T, n, FLOW, N)
		stochastics = [betahat[:, :, 0], vac_fun, test_fun, Movers]
		tune_tn1 = tune_t - ck[k] * hk
		tune_t1 = tune_t + ck[k] * hk
		boolt1 = tune_t1 <= 0
		boolt2 = tune_t1 >= 1
		tune_t1[boolt1] = 0.001
		tune_t1[boolt2] = 1
		booltn1 = tune_tn1 <= 0
		booltn2 = tune_tn1 >= 1
		tune_tn1[booltn1] = 0.001
		tune_tn1[booltn2] = 1
		hyperparameters_tn1 = hyperparameters.copy()
		hyperparameters_t1 = hyperparameters.copy()

		tparams_tn1 = []
		tparams_t1 = []
		vparams_tn1 = [tune_tn1[0]]
		vparams_t1 = [tune_t1[0]]
		COtn1 = run_sample_path(hyperparameters_tn1, stochastics, vaccine_policy, test_policy, vparams_tn1, tparams_tn1)
		COt1 = run_sample_path(hyperparameters_t1, stochastics, vaccine_policy, test_policy, vparams_t1, tparams_t1)

		ctn1 = COtn1.inst_list
		ct1 = COt1.inst_list

		cumulativetn1 = np.cumsum(ctn1)
		summation_meantn1 = np.cumsum(cumulativetn1)

		cumulativet1 = np.cumsum(ct1)
		summation_meant1 = np.cumsum(cumulativet1)

		Ftn1 = summation_meantn1[T-1]
		Ft1 = summation_meant1[T-1]

		G = (Ft1 - Ftn1)/(2*ck[k]) * hk

		dG += G
		dF += Ft1
	dG /= L
	dF /= L
	print('G = ' + str(dG))
	print('Ft = ' + str(dF))
	grad_squared = dG * dG
	mt = b1 * mt + (1-b1)*dG
	vt = b2 * vt + (1-b2)*grad_squared
	mhat = mt/(1-b1**(k+1))
	vhat = vt/(1-b2**(k+1))
	# g2 = 0.9 * g2 + 0.1 * dG *dG
	# alpha = et/g2
	# alphaG = np.min([0.1*np.ones(g2.shape), alpha * dG], axis=0)
	alpha = lr / grad_squared
	alphaG = alpha * dG
	tunable1 = tune_t - alphaG

	#project back into parameter space
	bool1 = tunable1 <= 0
	bool2 = tunable1 >= 1
	tunable1[bool1] = 0.001
	tunable1[bool2] = 1
	print('tunable1 = ' + str(tunable1))
	print('alpha = ' + str(alpha))
	print('mhat = ' + str(mhat))
	print('vhat = ' + str(vhat))
	print('ck = ' + str(ck[k]))
	tlist.append(list(tunable1))

	hyperparameters_test = hyperparameters.copy()
	tparams_test = []
	vparams_test = [tunable1[0]]
	COtest = run_sample_path(hyperparameters_test, stochastics, vaccine_policy, test_policy, vparams_test, tparams_test)

	ctest = COtest.inst_list

	cumulativetest = np.cumsum(ctest)
	summation_meantest = np.cumsum(cumulativetest)
	dFtest = summation_meantest[T-1]
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
	Fl.append(dFtest)

now = datetime.now()

current_time = now.strftime("%H,%M,%S")
dl.save_data(tlist, 'tuning_data/params_'+str(vac_tuner)+'_'+str(current_time)+'.obj')
dl.save_data(Fl, 'tuning_data/costs_'+str(vac_tuner)+'_'+str(current_time)+'.obj')
fig, ax = plt.subplots(1,2)
pl = np.array(tlist)
ax[0].plot(pl[:,0])
# ax[0].plot(pl[:,1])
# ax[0].plot(pl[:,2])
ax[1].plot(Fl[1:])

plt.show()






