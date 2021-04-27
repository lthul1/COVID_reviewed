import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import gurobipy as gp
from gurobipy import GRB
import utility

nvac = 500
nc = 5
N = np.int32(800 * np.random.rand(nc) + 400)
pI = 0.02 * np.random.rand(nc) + 0.1
pS = 1 - pI
pR = np.zeros(nc)
beta = 0.2 * np.random.rand(nc) + 0.7
gamma = 0.2

state = {'N':N, 'nc':nc, 'pS':pS,'pI':pI, 'pR':pR, 'beta':beta, 'gamma':gamma, 'nvac':nvac}

T = 25
alphas = {t:[-10e6] for t in range(T)}
betas = {t:[np.zeros(nc)] for t in range(T)}
alphas[T+1] = [0]
betas[T+1] = 0
for t in range(T+1):
	beta1 = beta + np.random.uniform(low=-0.05,high=0.05,size=nc)
	m = gp.Model("iP")
	# Create variables
	x = m.addMVar(shape=nc, vtype=GRB.CONTINUOUS, name="x")
	pI = state['pI']
	pS = state['pS']
	pR = state['pR']
	c = beta * pI
	m.setParam('OutputFlag', 0)
	m.setObjective(c @ x, GRB.MAXIMIZE)
	m.addConstr(np.ones(nc) @ x <= nvac, name="c")
	m.addConstr(0 <= x)
	m.addConstr(x <= (N-1)*pS)
	m.optimize()
	xtk = x.X
	pi = m.Pi
	Stxk = np.max([np.zeros(nc), (N-1)*pS - xtk], axis=0)
	St1 = (1-beta1*pI)*Stxk
	It1 = (1-gamma)*pI*N + (beta1*pI*Stxk)
	Rt1 = N*pR + gamma*pI*N + np.min([(N-1)*pS, xtk],axis=0)
	state['pS'] = St1/N
	state['pI'] = It1/N
	state['pR'] = Rt1/N
	# print(pS*N)
	# print(xtk)
	# print(Stxk)
	# print(St1)
	print(pi)

print(state['pS']*N)
print(np.sum(N))
# for t in np.arange(T,0,-1):




























