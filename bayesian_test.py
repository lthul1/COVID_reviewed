import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import gurobipy as gp
from gurobipy import GRB
import utility

from mpl_toolkits import mplot3d

nvac = 500
nc = 2
N = np.int32(800 * np.random.rand(nc) + 400)
pI = 0.02 * np.random.rand(nc) + 0.1
pS = 1 - pI
pR = np.zeros(nc)
beta = 0.2 * np.random.rand(nc) + 0.7
gamma = 0.2
lz = 0.3

state = {'N':N, 'nc':nc, 'pS':pS,'pI':pI, 'pR':pR, 'beta':beta, 'gamma':gamma, 'nvac':nvac}

def vac_pol(state):
	state1 = state.copy()
	Sbar = state1['pS'] * (state1['N']-1)
	c = state1['beta'] * state1['pI']
	m = gp.Model("iP")
	# Create variables
	x = m.addMVar(shape=nc, vtype=GRB.INTEGER, name="x")

	m.setParam('OutputFlag', 0)
	m.setObjective(c @ x, GRB.MAXIMIZE)
	m.addConstr(np.ones(nc) @ x <= nvac, name="c")
	m.addConstr(0 <= x)
	m.addConstr(x <= Sbar)
	m.optimize()
	return x.X

xvac = vac_pol(state)
print(xvac)

Sbar = pS * N
Ibar = pI * N
Rbar = pR * N

eps = 1e-6
EISx = N * pI * ((N - 1) * pS - xvac)
ESI = N * (N - 1) * pI * pS
ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
VarSI = ES2I2 - ESI ** 2
VarxI = xvac ** 2 * (N * pI * (1 - pI))
cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
s = np.sqrt(VarSI + VarxI + cov + eps)
mean = EISx

# E[I S^x]
term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

# E[max(0,S-x)] = E[S] - E[min(S,x)]
sigma_susc = np.sqrt(N*pS*(1-pS) + eps)
ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac)/sigma_susc) + sigma_susc*stats.norm.pdf((Sbar - xvac)/sigma_susc)
Emin = ESx - Sbar

# Get the predicted states at t+1
Sbar1 = ESx - term
Ibar1 = (1-gamma) * Ibar + term
Rbar1 = Rbar + gamma*Ibar + Emin

ntest = 100
stride = 10

XT1,XT2 = np.meshgrid(np.arange(0,ntest,stride), np.arange(0,ntest,stride))
vmean = []
vvar = []
for i in range(XT1.shape[0]):
	for j in range(XT1.shape[1]):
		Amean = 0
		Avar = 1
		nn = 15
		for l in range(nn):
			xtester = [XT1[i,j], XT2[i,j]]
			Ihat = [np.random.binomial(xtester[k], Ibar1[k]/N[k]) for k in range(nc)]
			p_inf = (Ihat + lz*Ibar1) / (xtester + lz * N)

			# project the other dimensions back onto it
			sv = np.array([utility.simplex_projector(np.array([Sbar1[i]/N[i], Rbar1[i]/N[i]]), const=p_inf[i]) for i in range(nc)])
			ps0 = sv[:,0]
			pr0 = sv[:,1]
			pi0 = p_inf
			state_temp = state.copy()
			state_temp['pS'] = ps0
			state_temp['pI'] = pi0
			state_temp['pR'] = pr0
			C = np.sum(-gamma * N * pi0 + beta * pi0 * ((N-1)*ps0 - vac_pol(state_temp)))
			Amean = (1/(l+1))*(l*Amean + C)
			if l == 0:
				Avar = (l/(l+1))*Avar
			else:
				Avar = (l / (l + 1)) * Avar + (1/l) * (C - Amean)**2

		vmean.append(Amean)
		vvar.append(Avar)

ax = plt.axes(projection='3d')
rs = np.int32(ntest/stride)
ax.plot_surface(XT1,XT2,np.array(vvar).reshape(rs,rs))

plt.show()




















