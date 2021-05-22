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
n = 25

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
nc = 53
T = 25
xi = 1
lz = 0.1
a = 0.9
b = 0.1
cc = 0.7
dd = 0.3
fn = 0
fp = 0.04
ir = 0.01*np.zeros(nc)
np.random.seed(1)
ii = np.random.randint(low=0,high=nc,size=10)
np.random.seed(None)
ir[ii] = 1
p_inf0 = 0.1 * ir
p_rec0 = 0.1
gamma_ = 0.15
alpha_ = 0.8
N = gen_N_NH(nc)
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

Rlist = []
Dlist = []
tests = np.arange(1,501,10)
for m in tests:
    print('test kits: '+str(m))
    # initialize process classes
    vac_proc = vaccine_process2(nc, T, N)
    test_proc = test_process2(nc, T, N)
    # create alias for the methods
    vac_fun = vac_proc.stoch
    test_fun = test_proc.shortage


    # initial controller state


    # list of vaccine policies
    vaccine_policies = [vac_policies.risk_DLA_param]
    mv = len(vaccine_policies)
    vac_names = ['DLA_param']
    # vparams0 is null policy params
    vparams0 = []
    vparams1 = [50, 10]
    vparams2 = [0.05]
    vparams3 = [0.85]
    vparams4 = [0.85, 0.4]
    vparams5 = [6, 100, 0.5]
    vparams6 = [.05, 4, 0.5, 2.75, .515, 240]

    vparam_list = [vparams6]

    # list of testing policies
    testing_policies = [test_policies.pure_exploration, test_policies.EI, test_policies.prop_greedy_trade, test_policies.prop_greedy_trade, test_policies.prop_greedy_trade, test_policies.prop_greedy_trade]
    mt = len(testing_policies)
    test_names = ['even',  'EI', 'trade_0.2', 'trade_0.4', 'trade_0.6', 'trade_0.8']
    # tparams0 is null policy params
    tparams0 = []
    tparams1 = [10]
    tparams2 = [6]
    tparams3 = [0.2]
    tparams4 = [0.4]
    tparams5 = [0.6]
    tparams6 = [0.8]
    tparam_list = [tparams0,  tparams0, tparams3, tparams4, tparams5, tparams6]
    blist = [False, False, False, False, False, False]


    betahat = gen_betas_NH(nc, T, n, 2)
    Movers = [gen_FLOW(nc, T, n, FLOW, N) for i in range(n)]
    rs = np.ones(n)
    COlist = [[[] for j in range(mv)] for i in range(mt)]
    Xvaclist = [[[] for j in range(mv)] for i in range(mt)]
    Inc = [[[] for j in range(mv)] for i in range(mt)]
    # seeds = np.random.randint(low=0, high=500, size=n)
    seeds = [None for _ in range(n)]
    for i in range(mt):
        for j in range(mv):
            for k in range(n):
                # stochastic list
                # stochastics[0] = betahat[k] - beta values for sample k
                # stochastics[1] = vac_fun - vaccine production function
                # stochastics[2] = test_fun - test kit production function
                print('k = '+str(k))
                start = datetime.now()
                stochastics = [betahat[:, :, k], vac_fun, test_fun, Movers[k], rs[k], seeds[k]]
                costs, xvaclist, Ilist = run_sample_path2(hyperparameters, stochastics, vaccine_policies[j], testing_policies[i],
                                        vparam_list[j], tparam_list[i], m, blist[i])
                print('Vaccine Policy: '+str(vac_names[j]) + '  , Test Policy: ' +str(test_names[i]) + '  time:  ' +str(datetime.now() - start))
                COlist[i][j].append(costs.inst_list)
                Xvaclist[i][j].append(xvaclist)
                Inc[i][j].append(Ilist)

    costs = COlist.copy()
    counter2 = 0
    Ntot = np.sum(N) * np.ones(T)
    for j in range(len(test_names)):
        for k in range(len(vac_names)):
            cumulative = np.cumsum(costs[j][k], axis=1)
            instant_mean = np.mean(costs[j][k], axis=0)
            cumulative_mean = np.mean(cumulative, axis=0)
            summation_mean = np.cumsum(cumulative_mean)
            if j == 0:
                Rlist.append(summation_mean[T-1])
            elif j==1:
                Dlist.append(summation_mean[T - 1])

plt.plot(tests, Rlist)
plt.plot(tests, Dlist)
plt.title('Testing Kits Available versus Performance')
plt.legend(['Even', 'MaxVar', 'Tradeoff_20', 'Tradeoff_40', 'Tradeoff_60', 'Tradeoff_80'])
plt.show()
