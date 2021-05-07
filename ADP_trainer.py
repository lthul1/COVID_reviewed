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

import gurobipy as gp
from gurobipy import GRB
import cost_object

# n = number of samples paths to simulation
n = 1000

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

def draw_states(n, hyperparameters):
    # n is the number of samples
    pS = 0.5*np.random.rand(nc,n)
    pI = 0.5*np.random.rand(nc,n)
    pR = 1 - pS - pI
    beta = np.random.uniform(low=0.5, high=0.95, size=[nc,n])
    nvac = np.random.randint(np.sum(np.floor(0.1*hyperparameters[14])),size=n)
    states = [{'pS':pS[:,i], 'pI':pI[:,i], 'pR':pR[:,i], 'nvac':nvac[i], 'beta':beta[:,i]} for i in range(n)]
    return states

def sim_trans(state0, xvac, hyperparameters):
    state = state0.copy()
    beta = state['beta']
    gamma = hyperparameters[12]
    eps = 10e-6
    N = hyperparameters[14]
    pS = state['pS']
    pI = state['pI']
    pR = state['pR']

    # compute the necessary satistics
    Sbar = pS * N
    Ibar = pI * N
    Rbar = pR * N

    EISx = N * pI * ((N - 1) * pS - xvac)
    ESI = N * (N - 1) * pI * pS
    ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
    VarSI = ES2I2 - ESI ** 2
    VarxI = xvac ** 2 * (N * pI * (1 - pI))
    cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
    s = np.sqrt(VarSI + VarxI + cov + eps)

    mean = EISx

    # E[I S^x]
    term = (beta / N) * (mean * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

    # E[max(0,S-x)] = E[S] - E[min(S,x)]
    sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
    ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
        (Sbar - xvac) / sigma_susc)
    Emin = Sbar - ESx

    # Get the predicted states at t+1
    Sbar1 = ESx - term
    Ibar1 = (1 - gamma) * Ibar + term
    Rbar1 = Rbar + gamma * Ibar + Emin

    return np.sum(Ibar1 - Ibar)

def sim_decision(state, v, hyperparameters):
    state0 = state.copy()
    xi = hyperparameters[2]
    beta = state0['beta']
    nvac = state0['nvac']
    N = hyperparameters[14]
    thetaS = v[:nc]
    thetaI = v[nc:2*nc]
    thetaR = v[2*nc:3*nc]
    pI = state0['pI']
    pS = state0['pS']
    pR = state0['pR']
    Ibar = pI * N
    Rbar = pR * N
    Sbar = pS * N

    c = -beta/N * xi * Ibar  + thetaS * xi * (beta/N * Ibar - 1) + thetaI * (-beta/N * Ibar * xi) + thetaR * xi
    m = gp.Model("iP")
    # Create variables
    x = m.addMVar(shape=hyperparameters[0], vtype=GRB.INTEGER, name="x")

    m.setParam('OutputFlag', 0)
    m.setObjective(c @ x, GRB.MAXIMIZE)
    m.addConstr(np.ones(nc) @ x <= nvac, name="c")
    m.addConstr(0 <= x)
    m.addConstr(x <= Sbar)
    m.optimize()
    return x.X


def getSvec(states):
    Svecs = np.zeros([3*nc+1, len(states)])
    for i in range(len(states)):
        statei = states[i].copy()
        pS = statei['pS']
        pI = statei['pI']
        pR = statei['pR']
        nvac = statei['nvac']
        Svecs[:nc,i] = pS
        Svecs[nc:2*nc,i] = pI
        Svecs[2*nc:3*nc,i] = pR
        Svecs[3*nc,i]= nvac
    return Svecs



def compV(statei, v):
    pS = statei['pS']
    pI = statei['pI']
    pR = statei['pR']
    nvac = statei['nvac']
    g = np.zeros(3*nc+1)
    g[:nc] = pS
    g[nc:2*nc] = pI
    g[2*nc:3*nc] = pR
    g[3*nc] = nvac
    return np.dot(g,v)



V = {t:np.random.rand(3*nc + 1) for t in np.arange(1, T+1)}
V[T] = np.zeros(3*nc+1)
for t in np.arange(T-1,0,-1):
    print('t = '+str(t))
    states = draw_states(n, hyperparameters)
    Svec = getSvec(states)
    vlist_t = []
    for i in range(n):
        statei = states[i].copy()
        xsimi = sim_decision(statei, V[t+1], hyperparameters)
        vti = sim_trans(statei, xsimi, hyperparameters) + compV(statei, V[t+1])
        vlist_t.append(vti)

    X = Svec.T
    nn = np.linalg.inv(np.dot(X.T, X))
    ny = np.dot(X.T, np.array(vlist_t))
    tstar = np.dot(nn, ny)
    V[t] = tstar

now = datetime.now()
curtime = now.strftime("%H,%M,%S")
dl.save_data(V, 'ADP/data_'+str(nc)+'_time_'+str(curtime)+'.obj')




plt.show()
